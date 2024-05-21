from torch import nn


class TEMP(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256):
        super(TEMP, self).__init__()
        add_dim = 128
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, add_dim),
            nn.BatchNorm1d(add_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(add_dim, 1)
        )

    def forward(self, x):
        return self.model(x)


class TEMPDepthClassifier(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256):
        super(TEMPDepthClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, 3),

            nn.Softmax(0)
        )

    def forward(self, x):
        return self.model(x)

