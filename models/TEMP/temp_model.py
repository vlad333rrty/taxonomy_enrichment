from torch import nn
from transformers import BertModel, BertTokenizer


class TEMP(nn.Module):
    def __init__(self):
        super(TEMP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
