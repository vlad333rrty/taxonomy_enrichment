from torch import nn
from transformers import BertModel, BertTokenizer


class TEMP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(TEMP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
