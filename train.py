import argparse
import torch
import os
import sys
import shutil
import numpy as np
import visdom
import criterion
import cv2
from model import MetricLearner
from dataset import MetricData, SourceSampler

def get_args():
    parser = argparse.ArgumentParser(description='Face Occlusion Regression')
    # train
    parser.add_argument('--pretrain', type=str, default='/root/.torch/models/googlenet-1378be20.pth', help='pretrain googLeNet model paht')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs plans to train in total')
    parser.add_argument('--epoch_start', type=int, default=0, help='start epoch to count from')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--ckpt', type=str, default='./ckpt', help='checkpoint folder')
    parser.add_argument('--resume', action='store_true', help='load previous best model and resume training')
    parser.add_argument('--num_workers', default=2, type=int, help='')
    # test
    parser.add_argument('--test', action='store_true', help='switch on test mode')
    # annotation
    parser.add_argument('--anno', type=str, required=True, help='location of annotation file')
    parser.add_argument('--anno_test', type=str, required=True, help='location of test data annotation file')
    parser.add_argument('--img_folder', type=str, required=True, help='folder of image files in annotation file')
    parser.add_argument('--img_folder_test', type=str, default='', help='folder of test image files in annotaion file')
    parser.add_argument('--idx_file', type=str, required=True, help='idx file for every label class')
    parser.add_argument('--idx_file_test', type=str, default='idx_file.pkl', help='idx file for test data, should be .pkl format')
    # model hyperparameter
    parser.add_argument('--in_size', type=int, default=128, help='input tensor shape to put into model')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
    data = MetricData(data_root=args.img_folder, anno_file=args.anno, idx_file=args.idx_file)
    dataset = torch.utils.data.DataLoader(data, batch_size=args.batch, sampler=SourceSampler(data, args.batch//2), drop_last=True, num_workers=args.num_workers)
    model = MetricLearner(pretrain=args.pretrain)
    if args.resume:
        if args.ckpt.endswith('.pth'):
            state_dict = torch.load(args.ckpt)
        else:
            state_dict = torch.load(os.path.join(args.ckpt, 'best_performance.pth'))
        best_performace = state_dict['loss']
        start_epoch = state_dict['epoch']
        model.load_state_dict(state_dict['state_dict'], strict=False)
        print('Resume training.')
    else:
        start_epoch = 0
        best_performace = np.Inf
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # TEST DATASET
    if args.test and args.resume:
        dataset_test = torch.utils.data.DataLoader(MetricData(args.img_folder_test, args.anno_test, args.idx_file_test, return_fn=True), \
                            batch_size=1, shuffle=True, drop_last=False, num_workers=max(1, int(args.num_workers/2)))
        vis = visdom.Visdom()
        model.eval()
        top_4 = {}
        with torch.no_grad():
            for i, batch in enumerate(dataset_test):
                batch[0] = batch[0].to(device)
                if i < 4:
                    query, atts = model(batch[0], ret_att=True)
                    imgs = MetricData.tensor2img(batch[0])
                    print('number of images in batch:', len(imgs), imgs[0].shape, imgs[0].min(), imgs[0].max())
                    print('number of attentions in batch:', len(atts), atts[0].shape)
                    vis.images(np.concatenate([cv2.resize(np.repeat(atts[i].cpu().numpy()[0, ...].mean(axis=0)[...,np.newaxis], 3, axis=-1), (224, 224))[np.newaxis]*255 for i in range(3)]), \
                        win=i+1000, opts=dict(title='Att_%d'%i))
                    top_4[i] = {'fn': batch[1][0], 'query': query.cpu().numpy(), 'top_8': []}
                    vis.image(np.transpose(cv2.imread(os.path.join(args.img_folder_test, top_4[i]['fn']))[..., ::-1], (2, 0, 1)), \
                        win=i+100, opts=dict(title='Query_%d'%i))    
                    print('Added query.')                
                else:
                    embedding = model(batch[0]).cpu().numpy()
                    for j in range(4):
                        dist = np.sum((top_4[j]['query'] - embedding)**2)
                        if len(top_4[j]['top_8']) < 8 or (len(top_4[j]['top_8']) >= 8 and dist < top_4[j]['top_8'][-1]['distance']):
                            top_4[j]['top_8'].append({'fn': batch[1][0], 'distance': dist})
                            if len(top_4[j]['top_8']) > 8:
                                last_fn = top_4[j]['top_8'][-1]['fn']
                                top_4[j]['top_8'] = sorted(top_4[j]['top_8'], key=lambda x: x['distance'])
                                print('%d Sorted.'%j, top_4[j]['top_8'])
                                top_4[j]['top_8'] = top_4[j]['top_8'][:8]
                                update = False
                                for d in top_4[j]['top_8']:
                                    if d['fn'] == last_fn:
                                        update = True
                                        print('\nUpdated\n')
                                        break
                                if update:
                                    imgs = np.concatenate([np.transpose(cv2.resize(cv2.imread(os.path.join(args.img_folder_test, d['fn'])), (250, 250))[..., ::-1], (2, 0, 1))[np.newaxis] for d in top_4[j]['top_8']])
                                    vis.images(imgs, win=j, nrow=2, opts=dict(title='IMG_%d'%j))

        for item in top_4.values():
            print(item['fn'], '\n', item['top_8'], '\n\n')
        sys.exit()

    for epoch in range(start_epoch, args.epochs):
        model.train()

        loss = 0
        for i, batch in enumerate(dataset):
            batch = batch.to(device)
            embeddings = model(batch)

            optimizer.zero_grad()
            l = criterion.loss_func(embeddings)
            l.backward()
            optimizer.step()

            loss += l
            # print('\tloss: %.4f'%(loss / (i+1)))
        loss /= (i+1)
        print('Batch %d\tloss:%.4f'%(epoch, loss))
        if loss < best_performace:
            best_performace = loss
            torch.save({'state_dict': model.cpu().state_dict(), 'epoch': epoch+1, 'loss': loss}, \
                        os.path.join(args.ckpt, '%d_ckpt.pth'%epoch))
            shutil.copy(os.path.join(args.ckpt, '%d_ckpt.pth'%epoch), os.path.join(args.ckpt, 'best_performance.pth'))
            print('Saved model.')
            model.to(device)
