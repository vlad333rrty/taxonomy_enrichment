from torch import nn, functional
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertOnlyMLMHead


class TaxoPrompt(nn.Module):
    def __init__(self, config):
        super(TaxoPrompt, self).__init__()
        self.__model = BertOnlyMLMHead(config)

    def forward(self, x):
        return self.__model(x)
