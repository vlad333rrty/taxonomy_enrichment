import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, BertTokenizerFast

from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider, create_synsets_batch
from src.taxo_expantion_methods.common.wn_dao import WordNetDao
from src.taxo_expantion_methods.utils.utils import get_synset_simple_name


class Inferer:
    def __init__(self, tokenizer: BertTokenizer, bert: BertModel, all_synsets):
        self.__tokenizer = tokenizer
        self.__bert = bert
        self.__all_synsets = all_synsets

    def __taxo_prompt_to_str(self, terms, parent_tokens):
        xs = []
        for i in range(len(terms)):
            xs.append('What is parent-of {}? It is {}'.format(terms[i].value().split('.')[0], '[MASK]' * len(parent_tokens)))

        base_t = self.__tokenizer(
            xs,
            padding=True,
            return_tensors='pt',
            add_special_tokens=True
        )['input_ids']
        return base_t

    def score(self, output, tokens, prompt):
        res = 0
        i = 0
        j = 0
        while j < len(prompt):
            if prompt[j] == self.__tokenizer.mask_token_id:
                res += output[j][tokens[i]]
                i += 1
            j += 1
        return res / len(tokens)

    def infer(self, model, terms, device):
        scores = {}
        for anchor in tqdm(self.__all_synsets):
            with torch.no_grad():
                tokens = self.__tokenizer(
                    get_synset_simple_name(anchor),
                    padding=True,
                    truncation=True,
                    add_special_tokens=False
                )['input_ids']
                prompts = self.__taxo_prompt_to_str(terms, tokens)
                outputs = self.__bert(
                    prompts.to(device),
                    output_hidden_states=True
                )
                output = model(outputs[0])
            i = 0
            for elem in output:
                score = self.score(elem, tokens, prompts[i])
                concept = terms[i].value()
                if concept not in scores:
                    scores[concept] = (score, anchor)
                elif scores[concept][0] < score:
                    scores[concept] = (score, anchor)
                i += 1
        return scores


class TaxoPromptTermInferencePerformerFactory:
    @staticmethod
    def create(device):
        wn_reader = WordNetDao.get_wn_30()
        all_synsets = SynsetsProvider.get_all_leaf_synsets_with_common_root(wn_reader.synset('food.n.01'))
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        return Inferer(tokenizer, bert_model, all_synsets)
