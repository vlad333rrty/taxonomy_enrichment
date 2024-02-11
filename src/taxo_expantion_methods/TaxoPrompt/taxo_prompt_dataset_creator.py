import random
import time

import torch
from tqdm import tqdm
from transformers import BertTokenizer

from src.taxo_expantion_methods.TaxoPrompt.random_walk import ExtendedTaxoGraphNode
from src.taxo_expantion_methods.utils.utils import get_synset_simple_name, paginate


class _TaxoPrompt:
    def __init__(self, concept, parent, taxonomic_context):
        self.concept = concept
        self.parent = parent
        self.taxonomic_context = taxonomic_context


class TaxoPromptBatch:
    def __init__(self, prompts, pdefs):
        self.prompts = prompts
        self.pdefs = pdefs


class TaxoPromtBuilder:
    def __init__(self):
        self.__buffer = []

    def add(self, value):
        self.__buffer.append(value)
        return self

    def set(self, index, value):
        self.__buffer[index] = value

    def __str__(self):
        return ' '.join(self.__buffer)


class TaxoPromptDsCreator:
    def __init__(self, tokenizer: BertTokenizer):
        self.__tokenizer = tokenizer

    def __build_taxonomic_context(self, concept_node: ExtendedTaxoGraphNode, limit):
        stack = [concept_node]
        used = set()
        used.add(concept_node)
        buffer = [concept_node.get_synset().name()]
        while len(stack) > 0 and len(used) < limit:
            node = stack.pop()
            not_visited = list(filter(lambda x: x[1] not in used, node.get_edges()))
            if len(not_visited) == 0:
                break
            random_edge = random.choice(not_visited)
            buffer.append(random_edge[0].value)
            buffer.append(random_edge[1].get_synset().name())
            stack.append(random_edge[1])
            used.add(node)

        return ' '.join(buffer)

    def __build_extended_taxonomic_context(self, concept_node, limit, tau):
        contexts = []
        for i in range(tau):
            contexts.append(self.__build_taxonomic_context(concept_node, limit))
        return '.'.join(contexts)

    def __build_taxo_prompt(self, concept_node: ExtendedTaxoGraphNode, limit, tau):
        concept = concept_node.get_synset()
        random_parent = random.choice(concept_node.get_parent_nodes()).get_synset()
        taxonomic_context = self.__build_extended_taxonomic_context(concept_node, limit, tau)
        return _TaxoPrompt(concept, random_parent, taxonomic_context)


    def set_mask(self, ids):
        it_t, is_t = self.__tokenizer.vocab['it'], self.__tokenizer.vocab['is']
        for i in range(len(ids)):
            if ids[i] == it_t and ids[i + 1] == is_t:
                break
        j = i + 2
        while j < len(ids):
            ids[j] = self.__tokenizer.mask_token_id
            j += 1
        return ids

    def __taxo_prompt_to_str(self, taxo_prompt):
        sep = '[SEP]'
        pdef = taxo_prompt.parent.definition()
        builder = TaxoPromtBuilder()
        (builder.add('what is parent-of')
         .add(get_synset_simple_name(taxo_prompt.concept))
         .add('?')
         .add('it is')
         .add(get_synset_simple_name(taxo_prompt.parent))
         .add(pdef)
         )
        base_str = builder.__str__()
        base_t = self.__tokenizer.encode_plus(
            base_str,
            padding=True,
            return_tensors='pt',
            add_special_tokens=False
        )['input_ids']
        masked_ids = self.set_mask(base_t.clone()[0])
        masked_str = self.__tokenizer.decode(masked_ids)
        p1 = f'{masked_str}{sep}{taxo_prompt.concept.definition()}'
        p2 = f'{masked_str}{sep}{taxo_prompt.taxonomic_context}'
        e1 = f'{base_str}{sep}{taxo_prompt.concept.definition()}'
        e2 = f'{base_str}{sep}{taxo_prompt.taxonomic_context}'
        return p1, e1, p2, e2

    def prepare_ds(self, train_nodes, relations_limit, tau, batch_size=8):
        samples = []
        for node in tqdm(train_nodes):
            taxo_prompt = self.__build_taxo_prompt(node, relations_limit, tau)
            p1, e1, p2, e2 = self.__taxo_prompt_to_str(taxo_prompt)
            samples.append((p1, e1, p2, e2))

        pointer = 0
        batches = []
        while pointer < len(samples):
            prompts = []
            pdefs = []
            i = 0
            while pointer < len(samples) and i < batch_size:
                prompts.append(samples[pointer][0])
                pdefs.append(samples[pointer][1])
                prompts.append(samples[pointer][2])
                pdefs.append(samples[pointer][3])
                pointer += 1
                i += 2
            batches.append(TaxoPromptBatch(prompts, pdefs))

        return batches
