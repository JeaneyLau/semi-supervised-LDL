import os
import torch
from torch.utils import data
from PIL import Image
import torch.nn as nn
import torchvision
import numpy as np
import operator
import itertools
import cv2
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

# 计算指标

def calculate_specificity_sensitivity(Y_true,Y_predict):
    confusion = confusion_matrix(Y_true,Y_predict)
    n_classes = confusion.shape[0]
    sensitivity = []
    specificity = []
    for i in range(n_classes):
        # 逐步获取 真阳，假阳，真阴，假阴四个指标，并计算三个参数
        ALL = np.sum(confusion)
        # 对角线上是正确预测的
        TP = confusion[i, i]
        # 列加和减去正确预测是该类的假阳
        FP = np.sum(confusion[:, i]) - TP
        # 行加和减去正确预测是该类的假阴
        FN = np.sum(confusion[i, :]) - TP
        # 全部减去前面三个就是真阴
        TN = ALL - TP - FP - FN
        sensitivity.append(TP/(TP+FN))
        specificity.append(TN/(TN+FP))
    return np.mean(sensitivity),np.mean(specificity)



class PublicSkinDataset_test(data.Dataset):

    def __init__(self,root,transform=None):
        super().__init__()
        fr = open(root)
        self.transform = transform
        self.BCdata = []
        self.BClabel = []  # prepare labels return
   
        for line in fr.readlines():
            line = line.strip('\n')
            listFromLine = line.split('  ')
            self.BCdata.append(listFromLine[0])
            self.BClabel.append(int(listFromLine[1]))

            
    def __getitem__(self, index):

        img_path="../data/ACNE04/JPEGImages/"+self.BCdata[index]
        label=self.BClabel[index]    
        img = Image.open(img_path) 
        img = self.transform(img)

        return img,label
    
    def __len__(self):
        return len(self.BCdata)
    

class PublicSkinDataset_distribution(data.Dataset):

    def __init__(self,root,transform=None):
        super().__init__()
        fr = open(root)
        self.transform = transform
        self.BCdata = []
        self.BClabel = []  # prepare labels return
        self.count = []
   
        for line in fr.readlines():
            line = line.strip('\n')
            listFromLine = line.split('  ')
            self.BCdata.append(listFromLine[0])
            self.BClabel.append(int(listFromLine[1]))
            self.count.append(int(listFromLine[2]))
#             print(int(listFromLine[1]))

            
    def __getitem__(self, index):
#         print(index)
 
        img_path="../data/ACNE04/JPEGImages/"+self.BCdata[index]
        label=self.BClabel[index]
        count = self.count[index]
        img = Image.open(img_path) 
        img = self.transform(img)

        return img,label,count
    
    def __len__(self):
#         return 58*10
        return 116*2
#         return len(self.BCdata)


# semi-supervised dataloader

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size
        
#         print()

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size
    
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class TransformTwice_fixMatch:
    def __init__(self, transform,transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        return out1, out2
    

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)   

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))