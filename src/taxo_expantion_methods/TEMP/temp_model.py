from torch import nn
from transformers import BertModel, BertTokenizer


class TEMP(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256):
        super(TEMP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x)
