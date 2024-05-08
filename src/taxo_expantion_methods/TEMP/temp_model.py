from torch import nn
from transformers import BertModel, BertTokenizer


class TEMP(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256):
        super(TEMP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
