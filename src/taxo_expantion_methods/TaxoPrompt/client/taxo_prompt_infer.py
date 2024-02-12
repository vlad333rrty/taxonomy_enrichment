import torch
from nltk.corpus import WordNetCorpusReader
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer, BertTokenizerFast, BertModel, BertConfig

from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider, create_synsets_batch
from src.taxo_expantion_methods.TaxoPrompt.taxo_prompt_dataset_creator import TaxoPromtBuilder
from src.taxo_expantion_methods.TaxoPrompt.taxo_prompt_model import TaxoPrompt
from src.taxo_expantion_methods.common.wn_dao import WordNetDao
from src.taxo_expantion_methods.utils.utils import get_synset_simple_name


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
        sep = '[SEP]'
        pdef = parent.definition()
        xs = []
        for i in range(len(concepts)):
            builder = TaxoPromtBuilder()
            (builder.add('what is parent-of')
             .add(concepts[i])
             .add('?')
             .add('it is')
             .add(get_synset_simple_name(parent))
             .add(pdef)
             .add(sep)
             .add(definitions[i])
             )
            xs.append(builder.__str__())

        base_t = self.__tokenizer.batch_encode_plus(
            xs,
            padding=True,
            return_tensors='pt',
            add_special_tokens=False
        )['input_ids']
        indices = []
        for elem in base_t:
            x = self.set_mask(elem)
            indices.append(x)
        return base_t, indices

    def score(self, output, definition, start_end):
        res = 0
        ks = output[start_end[0]]
        tokens = self.__tokenizer.encode_plus(
            definition,
            padding=True,
            return_tensors='pt',
            add_special_tokens=False
        )['input_ids']
        for token in tokens[0]:
            res += ks[token]
        return res / len(tokens[0])

    def infer(self, model, concepts, definitions, device):
        scores = {}
        for anchor in tqdm(self.__all_synsets):
            prompts, indices = self.__taxo_prompt_to_str(concepts, definitions, anchor)
            with torch.no_grad():
                outputs = self.__bert(
                    prompts.to(device),
                    output_hidden_states=True
                )
            output = model(outputs[0])
            i = 0
            for elem in output:
                score = self.score(elem, anchor.definition(), indices[i])
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
