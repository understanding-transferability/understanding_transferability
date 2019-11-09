import os
import numpy as np
import tensorpack
import time
import random
import numbers
from scipy.misc import imread, imresize
import tensorlayer as tl
from six.moves import cPickle
from wheel import *

import warnings
# disable warning of imread
warnings.filterwarnings('ignore', message='.*', category=Warning)


class CustomDataLoader(object):
    def __init__(self, dataset, batch_size, num_threads=8,remainder=None):
        self.ds0 = dataset
        self.batch_size = batch_size
        self.num_threads = num_threads
        
        if not remainder:
            try:
                is_train = self.ds0.is_train
                remainder = False if is_train else True # if is_train, there is no need to set reminder 
            except Exception as e:
                # self.ds0 maybe doesn't have is_train attribute, then it has no test mode, set remainder = False
                remainder = False
        
        # use_list=False, for each in data point, add a batch dimension (return in numpy array)
        self.ds1 = tensorpack.dataflow.BatchData(self.ds0, self.batch_size,remainder=remainder, use_list=False,) 
        
        # use 1 thread in test to avoid randomness (test should be deterministic)
        self.ds2 = tensorpack.dataflow.PrefetchDataZMQ(self.ds1, nr_proc=self.num_threads if not remainder else 1)
        
        # required by tensorlayer package
        self.ds2.reset_state()
    
    def generator(self):
        """
        :return: if ``self.ds0.get_data()`` returns N elements,
            then this function returns a generator, which yields N elements in total (actually, it depends on ``ds0.size()``)
        """
        return self.ds2.get_data()


class BaseDataset(tensorpack.dataflow.RNGDataFlow):
    def __init__(self, is_train=True, skip_pred=None, transform=None, sample_weight=None, auto_weight=False):
        self.is_train = is_train
        self.skip_pred = skip_pred or (lambda data, label, is_train : False)
        self.transform = transform or (lambda data, label, is_train : (data, label))
        self.sample_weight = sample_weight or (lambda data, label : 1.0)

        self.datas = []
        self.labels = []

        self._fill_data()

        if auto_weight:
            assert sample_weight is None, 'auto_weight and sample_weight are mutually exclusive!'
            counter = Counter(np.asarray(self.labels).flatten())
            for x in counter:
                counter[x] = 1.0 / counter[x]
            self.sample_weight = (lambda data, label : counter[np.asarray(label).flatten()[0]])

        self._post_init()

    def _fill_data(self):
        raise NotImplementedError("not implemented!")

    def _post_init(self):
        tmp = [[data, label]  for (data, label) in zip(self.datas, self.labels) if not self.skip_pred(data, label, self.is_train) ]
        self.datas = [x[0] for x in tmp]
        self.labels = [x[1] for x in tmp]

        if callable(self.sample_weight):
            self._weight = [self.sample_weight(x, y) for (x, y) in zip(self.datas, self.labels)]
        else:
            self._weight = self.sample_weight
        self._weight = np.asarray(self._weight, dtype=np.float32).reshape(-1)
        assert len(self._weight) == len(self.datas), 'dimension not match!'
        self._weight = self._weight / np.sum(self._weight)
        # if weight is almost uniform(max / min < 1.5),then we treat it as uniform distribution.
        # (non-uniform sampling is time-consuming)
        self.uniform_weight_flag = True if np.max(self._weight) / np.min(self._weight) < 1.5 else False

    def size(self):
        return len(self.datas)

    def _get_one_data(self, data, label):
        raise NotImplementedError("not implemented!")

    def get_data(self):
        size = self.size()
        ids = list(range(size))
        # if weight is uniform, we only need to shuffle once and get data in sequence.
        if self.uniform_weight_flag:
            random.shuffle(ids)
        for _ in range(size):
            if not self.is_train:
                id = _
            else:
                if self.uniform_weight_flag:
                    id = ids[_]
                else:
                    id = np.random.choice(ids, p=self._weight)
            data, label = self._get_one_data(self.datas[id], self.labels[id])
            data, label = self.transform(data, label, self.is_train)
            yield np.asarray(data), np.asarray([label]) if isinstance(label, numbers.Number) else label


class TestDataset(BaseDataset):
    def __init__(self, N=100,  is_train=True, skip_pred=None, transform=None, sample_weight=None, auto_weight=False):
        self.N = N
        super(TestDataset, self).__init__(is_train=is_train, skip_pred=skip_pred, transform=transform, sample_weight=sample_weight, auto_weight=auto_weight)

    def _fill_data(self):
        self.datas = [[i, i + 1] for i in range(self.N)]
        self.labels = [2 * i + 1 for i in range(self.N)]

    def _get_one_data(self, data, label):
        return data, label


class CombinedDataset(BaseDataset):
    def __init__(self, datasets, weights):
        self.datasets = datasets
        self.weights = weights
        super(CombinedDataset, self).__init__(is_train=True, skip_pred=None, transform=None, auto_weight=False)

    def _fill_data(self):
        self.datas = sum([x.datas for x in self.datasets], [])
        self.labels = sum([x.labels for x in self.datasets], [])

        # make weights a probability distribution
        self.weights = np.asarray(self.weights, dtype=np.float32)
        self.weights = self.weights / np.sum(self.weights)

        self.iters = [x.get_data() for x in self.datasets]
        self.indexes = np.asarray(list(range(len(self.datasets))), dtype=np.int)

    def _get_one_data(self, data, label):
        index = np.random.choice(self.indexes, p=self.weights)
        try:
            return next(self.iters[index])
        except StopIteration as e:
            self.iters[index] = self.datasets[index].get_data()
            return next(self.iters[index])


class BaseImageDataset(BaseDataset):
    def __init__(self, imsize=224, is_train=True, skip_pred=None, transform=None, sample_weight=None, auto_weight=False):
        self.imsize = imsize
        super(BaseImageDataset, self).__init__(is_train, skip_pred, transform, sample_weight=sample_weight, auto_weight=auto_weight)

    def _get_one_data(self, data, label):
        im = imread(data, mode='RGB')
        if self.imsize:
            im = imresize(im, (self.imsize, self.imsize))
        return im, label


def one_hot(n_class, index):
    tmp = np.zeros((n_class,), dtype=np.float32)
    tmp[index] = 1.0
    return tmp


from collections import Counter
class FileListDataset(BaseImageDataset):
    def __init__(self, list_path, path_prefix='', imsize=224, is_train=True, skip_pred=None, transform=None, sample_weight=None, auto_weight=False):
        self.list_path = list_path
        self.path_prefix = path_prefix

        super(FileListDataset, self).__init__(imsize=imsize, is_train=is_train, skip_pred=skip_pred, transform=transform, sample_weight=sample_weight, auto_weight=auto_weight)

    def _fill_data(self):
        with open(self.list_path, 'r') as f:
            data = [[line.split()[0], line.split()[1] if len(line.split()) > 1 else '0'] for line in f.readlines() if line.strip()] # avoid empty lines
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1]) for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is space in image path?')
                raise e

class BaseDigitDataset(BaseDataset):

    def __init__(self, imsize=28, is_train=True, skip_pred=None, transform=None, sample_weight=None,
                 auto_weight=False):
        self.imsize = imsize
        super(BaseDigitDataset, self).__init__(is_train, skip_pred, transform, sample_weight=sample_weight,
                                               auto_weight=auto_weight)
    def _get_one_data(self, data, label):
        im = imread(data, mode='L')
        if self.imsize:
            im = imresize(im, (self.imsize, self.imsize))
        return im, label


class DigitFileListDataset(BaseDigitDataset):
    def __init__(self, list_path, path_prefix='', imsize=28, is_train=True, skip_pred=None, transform=None,
                 sample_weight=None, auto_weight=False):
        
        self.list_path = list_path
        self.path_prefix = path_prefix

        super(DigitFileListDataset, self).__init__(imsize=imsize, is_train=is_train, skip_pred=skip_pred, transform=transform,
                                              sample_weight=sample_weight, auto_weight=auto_weight)

    def _fill_data(self):
        with open(self.list_path, 'r') as f:
            data = [[line.split()[0], line.split()[1] if len(line.split()) > 1 else '0'] for line in f.readlines() if
                    line.strip()]  # avoid empty lines
            self.datas = [join_path(self.path_prefix, x[0]) for x in data]
            try:
                self.labels = [int(x[1]) for x in data]
            except ValueError as e:
                print('invalid label number, maybe there is space in image path?')
                raise e


