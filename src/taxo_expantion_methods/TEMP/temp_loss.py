import torch
from torch import nn


class TEMPLoss(nn.Module):
    def __init__(self, k):
        super(TEMPLoss, self).__init__()
        self.__k = k

    def __gamma(self, p1, p2):
        p1 = set(p1)
        p2 = set(p2)
        intersection = len(p1.intersection(p2))
        if intersection == 0:
            print('ERROR: NO COMMON SYNSETS FOR {} AND {}'.format(p1, p2))
        return (len(p1.union(p2)) / len(p1.intersection(p2)) - 1) * self.__k

    def forward(self, positive_paths, negative_paths, positive_outputs, negative_outputs):
        r = 0
        for i in range(len(positive_paths)):
            for j in range(len(negative_paths)):
                r += max(0, negative_outputs[j] - positive_outputs[i] + self.__gamma(positive_paths[i], negative_paths[j]))
        return r
