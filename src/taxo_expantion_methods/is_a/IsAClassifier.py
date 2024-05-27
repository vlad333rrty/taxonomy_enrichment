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

    def forward(self, possible_hypernym_embedding, node_embedding):
        extended_embedding = torch.concat([possible_hypernym_embedding, node_embedding])
        return self.model(extended_embedding)