import torch
from torch import nn


class IsAClassifier(nn.Module):
    def __init__(self):
        super(IsAClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(512, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(32, 1),
            nn.ReLU()
        )

    def forward(self, embeddings):
        return self.model(embeddings)

class IsALoss(nn.Module):
    def __init__(self, batch_size, device):
        super(IsALoss, self).__init__()
        self.__loss = nn.MarginRankingLoss(margin=1.0)
        self.__target = torch.ones(batch_size).to(device)

    def forward(self, positive, negative):
        return self.__loss(positive.view(-1), negative.view(-1), self.__target)
