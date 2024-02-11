from torch import nn
from torch.nn import CrossEntropyLoss


class MLMLoss(nn.Module):
    def __init__(self, config):
        super(MLMLoss, self).__init__()
        self.__config = config
        self.__loss = CrossEntropyLoss()

    def forward(self, scores, labels):
        return self.__loss(scores.view(-1, self.__config.vocab_size), labels.view(-1))
