import random
import time

import torch
from tqdm import tqdm
from transformers import BertTokenizer

from src.taxo_expantion_methods.TaxoPrompt.random_walk import ExtendedTaxoGraphNode
from src.taxo_expantion_methods.TaxoPrompt.terms_relation import Relation
from src.taxo_expantion_methods.utils.utils import get_synset_simple_name, paginate


class _TaxoPrompt:
    def __init__(self, concept, parent, taxonomic_context, masked_taxonomic_context):
        self.concept = concept
        self.parent = parent
        self.taxonomic_context = taxonomic_context
        self.masked_taxonomic_context = masked_taxonomic_context


class TaxoPromptBatch:
    def __init__(self, prompts, pdefs):
        self.prompts = prompts
        self.pdefs = pdefs



class TaxoPromptDsCreator:
    def __init__(self, tokenizer: BertTokenizer):
        self.__tokenizer = tokenizer

    def __build_taxonomic_context(self, concept_node: ExtendedTaxoGraphNode, limit):
        stack = [concept_node]
        used = set()
        used.add(concept_node)
        buffer = [get_synset_simple_name(concept_node.get_synset())]
        masked_buffer = [get_synset_simple_name(concept_node.get_synset())]
        while len(stack) > 0 and len(used) < limit:
            node = stack.pop()
            not_visited = list(filter(lambda x: x[1] not in used, node.get_edges()))
            if len(not_visited) == 0:
                break
            random_edge = random.choice(not_visited)
            buffer.append(random_edge[0].value)
            buffer.append(get_synset_simple_name(random_edge[1].get_synset()))

            masked_buffer.append('[MASK]')
            masked_buffer.append(get_synset_simple_name(random_edge[1].get_synset()))

            stack.append(random_edge[1])
            used.add(node)

        return ' '.join(buffer), ' '.join(masked_buffer)

    def __build_extended_taxonomic_context(self, concept_node, limit, tau):
        contexts = []
        masked_contexts = []
        for i in range(tau):
            buffer, masked_buffer = self.__build_taxonomic_context(concept_node, limit)
            contexts.append(buffer)
            masked_contexts.append(masked_buffer)
        return '.'.join(contexts), '.'.join(masked_contexts)

    def __build_taxo_prompt(self, concept_node: ExtendedTaxoGraphNode, limit, tau):
        concept = concept_node.get_synset()
        random_parent = random.choice(concept_node.get_parent_nodes()).get_synset()
        taxonomic_context, masked_context = self.__build_extended_taxonomic_context(concept_node, limit, tau)
        return _TaxoPrompt(concept, random_parent, taxonomic_context, masked_context)


    # def set_mask(self, ids):
    #     it_t, is_t = self.__tokenizer.vocab['it'], self.__tokenizer.vocab['is']
    #     for i in range(len(ids)):
    #         if ids[i] == it_t and ids[i + 1] == is_t:
    #             break
    #     j = i + 2
    #     while j < len(ids):
    #         ids[j] = self.__tokenizer.mask_token_id
    #         j += 1
    #     return ids

    def __taxo_prompt_to_str(self, taxo_prompt):
        sep = '[SEP]'
        base_str = f'What is {Relation.PARENT_OF.value} {get_synset_simple_name(taxo_prompt.concept)}? it is {get_synset_simple_name(taxo_prompt.parent)}'
        bas_str_mask = f'What is {Relation.PARENT_OF.value} {get_synset_simple_name(taxo_prompt.concept)}? it is [MASK]'

        pdef = taxo_prompt.parent.definition()
        ids = self.__tokenizer.encode_plus(
            pdef,
            padding=True,
            truncation=True,
            add_special_tokens=False
        )['input_ids']


        p1 = f'{bas_str_mask}{sep}{taxo_prompt.concept.definition()}'
        p2 = f'{bas_str_mask}{sep}{taxo_prompt.masked_taxonomic_context}'
        p_res = ' '.join([p1, '[MASK]' * len(ids), p2])
        e1 = f'{base_str}{sep}{taxo_prompt.concept.definition()}'
        e2 = f'{base_str}{sep}{taxo_prompt.taxonomic_context}'
        e_res = ' '.join([e1, pdef, e2])
        return p_res, e_res

    def prepare_ds(self, train_nodes, relations_limit, tau, batch_size=8):
        samples = []
        for node in tqdm(train_nodes):
            taxo_prompt = self.__build_taxo_prompt(node, relations_limit, tau)
            p, e = self.__taxo_prompt_to_str(taxo_prompt)
            samples.append((p, e))

        pointer = 0
        batches = []
        while pointer < len(samples):
            prompts = []
            pdefs = []
            i = 0
            while pointer < len(samples) and i < batch_size:
                prompts.append(samples[pointer][0])
                pdefs.append(samples[pointer][1])
                pointer += 1
                i += 1
            batches.append(TaxoPromptBatch(prompts, pdefs))

        return batches
