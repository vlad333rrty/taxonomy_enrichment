import torch
from torch import nn


class TEMPLoss(nn.Module):
    def __init__(self, k):
        super(TEMPLoss, self).__init__()
        self.__k = k

    def __gamma(self, p1, p2):
        p1 = set(p1)
        p2 = set(p2)
        return (len(p1.union(p2)) / len(p1.intersection(p2)) - 1) * self.__k

    def forward(self, positive_paths, negative_paths, positive_outputs, negative_outputs):
        r = 0
        for i in range(len(positive_paths)):
            for j in range(len(negative_paths)):
                r += max(0, negative_outputs[j] - positive_outputs[i] + self.__gamma(positive_paths[i], negative_paths[j]))
        return r


class TEMPDepthCalssifierLoss(nn.Module):
    def __init__(self):
        super(TEMPDepthCalssifierLoss, self).__init__()

    def forward(self, positive_paths, negative_paths, positive_outputs, negative_outputs):
        targets = []
        for _ in positive_paths:
            targets += [0, 1, 0]
        for i in range(len(negative_paths)):
            p = positive_paths[i]
            n = negative_paths[i]
            tensor = [1, 0 ,0] if len(n) < len(p) else [0, 0, 1]
            targets += tensor

        targets = torch.tensor(targets).view((64, 3))
        return nn.NLLLoss(
            torch.stack((positive_outputs, negative_outputs)),
            targets
        )