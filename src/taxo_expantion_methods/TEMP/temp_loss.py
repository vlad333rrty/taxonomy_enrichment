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


class TEMPLossExtended(TEMPLoss):
    def __init__(self, k, h):
        super().__init__(k)
        self.__h = h

    def forward(self, positive_paths, negative_paths, positive_outputs, negative_outputs):
        r = super().forward(positive_paths, negative_paths, positive_outputs, negative_outputs)
        accum = 0
        for i in range(len(positive_paths)):
            accum += abs(positive_paths[i].min_depth() - negative_paths[i].min_depth())
        return r + accum * self.__h
