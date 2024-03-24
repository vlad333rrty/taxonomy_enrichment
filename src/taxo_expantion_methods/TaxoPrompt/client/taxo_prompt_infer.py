import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider, create_synsets_batch
from src.taxo_expantion_methods.TaxoPrompt.terms_relation import Relation
from src.taxo_expantion_methods.common.wn_dao import WordNetDao


class Inferer:
    def __init__(self, tokenizer: BertTokenizer, bert: BertModel, all_synsets):
        self.__tokenizer = tokenizer
        self.__bert = bert
        self.__all_synsets = all_synsets

    def set_mask(self, ids):
        it_t, is_t = self.__tokenizer.vocab['it'], self.__tokenizer.vocab['is']
        for i in range(len(ids)):
            if ids[i] == it_t and ids[i + 1] == is_t:
                break
        j = i + 2
        start = j
        while j < ids[j] != self.__tokenizer.sep_token_id:
            ids[j] = self.__tokenizer.mask_token_id
            j += 1
        return (start, j)

    def __taxo_prompt_to_str(self, concepts, definitions, parent):
        pdef = parent.definition()
        xs = []
        for i in range(len(concepts)):
            descr = definitions[i]
            xs.append(' '.join([descr, '[MASK]' * len(pdef)]))

        base_t = self.__tokenizer.batch_encode_plus(
            xs,
            padding=True,
            return_tensors='pt',
        )['input_ids']
        return base_t

    def score(self, output, tokens, prompt):
        for i in range(len(prompt) - 2, 0, -1):
            if prompt[i] != self.__tokenizer.mask_token_id:
                break
        i = i + 1
        res = 0
        for j in range(len(tokens)):
            res += output[i][tokens[j]]
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
                prompts = self.__taxo_prompt_to_str(concepts, definitions, anchor)
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
        all_synsets = SynsetsProvider.get_all_synsets_with_common_root(wn_reader.synset('entity.n.01'))
        synsets_batches = create_synsets_batch(all_synsets, batch_size)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        return Inferer(tokenizer, bert_model, all_synsets)
