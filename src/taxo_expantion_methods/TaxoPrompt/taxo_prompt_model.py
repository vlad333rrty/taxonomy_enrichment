from torch import nn, functional


class TaxoPrompt(nn.Module):
    def __init__(self, h_dim, vocab_size):
        super(TaxoPrompt, self).__init__()
        self.l1 = nn.Linear(h_dim, h_dim)
        self.l2 = nn.Linear(h_dim, vocab_size)
        self.layer_norm = nn.LayerNorm(h_dim)

    def forward(self, bert_embedding):
        x = self.l1(bert_embedding)
        x = functional.F.relu(x)
        x = self.layer_norm(x)
        x = self.l2(x)
        return x
