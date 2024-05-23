import torch
from torch import nn


class TEMPLoss(nn.Module):
    def __init__(self, k, device):
        super(TEMPLoss, self).__init__()
        self.__k = k
        self.loss = nn.CrossEntropyLoss()
        self.__device = device

    def __gamma(self, p1, p2):
        p1 = set(p1)
        p2 = set(p2)
        return (len(p1.union(p2)) / len(p1.intersection(p2)) - 1) * self.__k

    def forward(self, positive_paths, negative_paths, positive_outputs, negative_outputs, outputs):
        r = 0
        for i in range(len(positive_paths)):
            for j in range(len(negative_paths)):
                r += max(0, negative_outputs[j][0] - positive_outputs[i][0] + self.__gamma(positive_paths[i], negative_paths[j]))
        # 2
        targets = []
        for _ in positive_paths:
            targets += [0., 1., 0.]
        for i in range(len(negative_paths)):
            p = positive_paths[i]
            n = negative_paths[i]
            tensor = [1., 0., 0.] if len(n) < len(p) else [0., 0., 1.]
            targets += tensor

        outputs_view = outputs[:, 1:]
        targets = torch.tensor(targets).view(outputs_view.size()).to(self.__device)
        return r + self.loss(outputs_view, targets)

class TEMPDepthCalssifierLoss(nn.Module):
    def __init__(self, device):
        super(TEMPDepthCalssifierLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.__device = device

    def forward(self, positive_paths, negative_paths, outputs):
        targets = []
        for _ in positive_paths:
            targets += [0., 1., 0.]
        for i in range(len(negative_paths)):
            p = positive_paths[i]
            n = negative_paths[i]
            tensor = [1., 0., 0.] if len(n) < len(p) else [0., 0., 1.]
            targets += tensor

        targets = torch.tensor(targets).view(outputs.size()).to(self.__device)
        return self.loss(outputs, targets)