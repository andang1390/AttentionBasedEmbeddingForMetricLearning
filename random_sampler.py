from torch.utils.data.sampler import Sampler
import torch

# From
# https://discuss.pytorch.org/t/random-sampler-implementation/18934/7
class RandomSampler(Sampler):
    r"""Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        self.i = 0
        self.idx = torch.randperm (len (self.data_source)).tolist ()

    def __iter__(self):
        self.i = 0
        while (self.i+1)*self.batch_size < len(self.data_source):
            yield self.idx[self.i*self.batch_size: (self.i+1)*self.batch_size]
            self.i+=1

    def __len__(self):
        return len(self.data_source)


class BatchSampler(Sampler):

    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for _, idx in enumerate(iter(self.sampler)):
            batch = idx
            yield batch

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        return len(self.sampler) // self.batch_size
