import os

import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM

from src.taxo_expantion_methods.utils.utils import paginate


class TaxoPromptTrainer:
    def __init__(self, tokenizer: BertTokenizer, bert: BertForMaskedLM, optimizer, checkpoint_save_path):
        self.__tokenizer = tokenizer
        self.__bert = bert
        self.__optimizer = optimizer
        self.__checkpoint_save_path = checkpoint_save_path

    def __save_checkpoint(self, epoch):
        save_path = os.path.join(self.__checkpoint_save_path, 'taxo_prompt_model_epoch_{}'.format(epoch))
        torch.save(self.__bert.state_dict(), save_path)

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

            output = self.__bert(
                tokens[:prompts_count],
                output_hidden_states=True,
                labels=tokens[prompts_count:],
                attention_mask=attention[:prompts_count]
            )
            loss = output.loss
            loss.backward()
            self.__optimizer.step()

            if i % 50 == 0:
                print(loss.item())
            if i % 1000:
                self.__save_checkpoint(epoch)

            # train_progess_monitor.step(model, epoch, batch_num, len(train_loader), loss, loss_fn)

    def train(self, train_data, device, epochs):
        ds_batches = paginate(train_data, epochs)
        for epoch in range(epochs):
            self.__train_epoch(ds_batches[epoch], device, epoch)
            self.__save_checkpoint(epoch)
