import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider, create_synsets_batch
from src.taxo_expantion_methods.common.wn_dao import WordNetDao


class Inferer:
    def __init__(self, tokenizer: BertTokenizer, bert: BertModel, all_synsets):
        self.__tokenizer = tokenizer
        self.__bert = bert
        self.__all_synsets = all_synsets

    def __taxo_prompt_to_str(self, definitions, parent_tokens):
        xs = []
        for i in range(len(definitions)):
            descr = definitions[i]
            xs.append(' '.join([descr, '[MASK]' * len(parent_tokens)]))

        base_t = self.__tokenizer.batch_encode_plus(
            xs,
            padding=True,
            return_tensors='pt',
        )['input_ids']
        return base_t

    def score(self, output, tokens, prompt):
        res = 0
        j = len(tokens) - 1
        for i in range(len(prompt) - 2, 0, -1):
            if prompt[i] != self.__tokenizer.mask_token_id:
                break
            res += output[i][tokens[j]]
            j -= 1
        return res / len(tokens)

    def infer(self, model, concepts, definitions, device):
        scores = {}
        for anchor in tqdm(self.__all_synsets):
            with torch.no_grad():
                tokens = self.__tokenizer.encode_plus(
                    anchor.definition(),
                    padding=True,
                    truncation=True,
                    add_special_tokens=False
                )['input_ids']
                prompts = self.__taxo_prompt_to_str(definitions, tokens)
                outputs = self.__bert(
                    prompts.to(device),
                    output_hidden_states=True
                )
                output = model(outputs[0])
            i = 0
            for elem in output:
                score = self.score(elem, tokens, prompts[i])
                concept = concepts[i]
                if concept not in scores:
                    scores[concept] = (score, anchor)
                elif scores[concept][0] < score:
                    scores[concept] = (score, anchor)
                i += 1
        return scores


class TaxoPromptTermInferencePerformerFactory:
    @staticmethod
    def create(device, batch_size):
        wn_reader = WordNetDao.get_wn_20()
        all_synsets = SynsetsProvider.get_all_leaf_synsets_with_common_root(wn_reader.synset('entity.n.01'))
        synsets_batches = create_synsets_batch(all_synsets, batch_size)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        return Inferer(tokenizer, bert_model, all_synsets)
