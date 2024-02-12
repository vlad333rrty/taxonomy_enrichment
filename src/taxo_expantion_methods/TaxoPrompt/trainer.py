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
            inputs = self.__tokenizer.batch_encode_plus(
                prompts + pdefs,
                padding=True,
                return_tensors='pt',
                truncation=True
            )
            tokens = inputs['input_ids'].to(device)
            attention = inputs['attention_mask'].to(device)
            prompts_count = len(prompts)

            with torch.no_grad():
                outputs = self.__bert(
                    tokens[:prompts_count],
                    output_hidden_states=True,
                    attention_mask=attention[:prompts_count]
                )
            x = self.__model(outputs[0])
            loss = self.__loss(x, tokens[prompts_count:])
            loss.backward()
            self.__optimizer.step()

            if i % 100 == 0:
                print(loss.item())

            # train_progess_monitor.step(model, epoch, batch_num, len(train_loader), loss, loss_fn)

    def train(self, train_data, device, epochs):
        for epoch in range(epochs):
            self.__train_epoch(train_data, device, epoch)
            self.__save_checkpoint(epoch)
