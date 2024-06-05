import os
import random

import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, BertModel

from src.taxo_expantion_methods.utils.utils import paginate


class TaxoPromptTrainer:
    def __init__(self, tokenizer: BertTokenizer, bert: BertModel, model, loss, optimizer, checkpoint_save_path):
        self.__tokenizer = tokenizer
        self.__bert = bert
        self.__model = model
        self.__loss = loss
        self.__optimizer = optimizer
        self.__checkpoint_save_path = checkpoint_save_path

    def __save_checkpoint(self, epoch):
        save_path = os.path.join(self.__checkpoint_save_path, 'taxo_prompt_model_epoch_{}'.format(epoch))
        torch.save(self.__model.state_dict(), save_path)

    def __tokenize(self, sentences, device):
        inputs = self.__tokenizer.batch_encode_plus(
            sentences,
            padding=True,
            truncation=True,
            return_tensors='pt',
            add_special_tokens=True
        )
        return inputs['input_ids'].to(device), inputs['attention_mask'].to(device)

    def __train_epoch(self, train_data, device, epoch):
        for i, batch in (pbar := tqdm(enumerate(train_data))):
            batch_num = i + 1
            pbar.set_description(f'EPOCH: {epoch}, BATCH: {batch_num}/{len(train_data)}')

            self.__optimizer.zero_grad()

            prompts = batch.prompts
            pdefs = batch.pdefs
            inputs_prompts = self.__tokenizer(
                prompts,
                padding=True,
                return_tensors='pt',
                truncation=True,
                add_special_tokens=False
            )
            inputs_pdefs = self.__tokenizer(
                pdefs,
                padding=True,
                return_tensors='pt',
                truncation=True,
                add_special_tokens=False
            )
            tokens = inputs_prompts['input_ids'].to(device)
            attention = inputs_prompts['attention_mask'].to(device)

            expected_tokens = inputs_pdefs['input_ids'].to(device)
            expected_attention = inputs_pdefs['attention_mask'].to(device)

            with torch.no_grad():
                outputs = self.__bert(
                    tokens,
                    output_hidden_states=True,
                    attention_mask=attention
                )
                outputs_expected = self.__bert(
                    expected_tokens,
                    output_hidden_states=True,
                    attention_mask=expected_attention
                )
            x = self.__model(outputs[0])
            y = self.__model(outputs_expected[0])
            loss = self.__loss(x, y)
            loss.backward()
            self.__optimizer.step()

            if i % 500 == 0 or batch_num == len(train_data):
                print(loss.item())

            # train_progess_monitor.step(model, epoch, batch_num, len(train_loader), loss, loss_fn)

    def train(self, train_data, device, epochs):
        for epoch in range(epochs):
            self.__train_epoch(train_data, device, epoch)
            self.__save_checkpoint(epoch)
