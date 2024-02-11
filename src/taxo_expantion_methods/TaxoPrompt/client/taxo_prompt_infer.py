import torch
from transformers import BertForMaskedLM, BertTokenizer, BertTokenizerFast


def infer(concept, definition, model: BertForMaskedLM, tokenizer: BertTokenizer):
    mask = '[MASK]'
    p1 = f'parent-of {concept} is {mask} [SEP] {definition}'
    target = f'parent-of {concept} is compound.n.01 [SEP] {definition}'
    inputs = tokenizer.batch_encode_plus(
        [target],
        padding=True,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    ids = inputs['input_ids']
    att = inputs['attention_mask']
    labels = ids.clone()
    ids[0][6] = tokenizer.mask_token_id
    outputs = model(
        ids,
        attention_mask=att,
        output_hidden_states=True,
        labels=labels
    )
    a = outputs



tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
c = 'phosphor'
d = 'substance that exhibits the phenomenon of luminescence'
infer(c, d, bert_model, tokenizer)