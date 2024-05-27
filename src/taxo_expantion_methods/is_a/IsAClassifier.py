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
            nn.Sigmoid()
        )

    def forward(self, embeddings):
        return self.model(embeddings)

class IsALoss(nn.Module):
    def __init__(self, batch_size, device):
        super(IsALoss, self).__init__()
        self.__loss = nn.BCELoss()
        n = batch_size // 2
        self.__target = torch.cat([torch.ones(n), torch.zeros(n)]).to(device)

    def forward(self, input):
        return self.__loss(input.view(-1), self.__target)
