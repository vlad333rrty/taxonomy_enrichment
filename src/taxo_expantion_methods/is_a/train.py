import os

import torch
from tqdm import tqdm

from src.taxo_expantion_methods.common.plot_monitor import Metric, PlotMonitor


class TrainProgressMonitor:
    def __init__(self, interval: int, valid_loader, epochs: int, plot_monitor):
        self.__interval = interval
        self.__running_loss = 0
        self.__running_items = 0
        self.__valid_loader = valid_loader
        self.__epochs = epochs
        self.__plot_monitor = plot_monitor


    def step(self, model, epoch, i, samples, loss, loss_fn, calc_val_loss=False):
        self.__plot_monitor.accept(Metric('Train loss', loss.item()))
        self.__running_loss += loss.item()
        self.__running_items += 1
        if i % self.__interval == 0 or i == samples:
            print(f'Epoch [{epoch + 1}/{self.__epochs}]. '
                  f'Batch [{i}/{samples}].'
                  f'Loss: {self.__running_loss / self.__running_items:.3f}. ')
            self.__plot_monitor.plot()



class IsATrainer:
    def __init__(self, embedding_provider, checkpoint_save_path):
        self.__embedding_provider = embedding_provider
        self.__checkpoint_save_path = checkpoint_save_path

    def __train_epoch(self, model, loss_fn, optimizer, train_loader, epoch,
                      train_progess_monitor: TrainProgressMonitor):
        for i, batch in (pbar := tqdm(enumerate(train_loader))):
            batch_num = i + 1
            pbar.set_description(f'EPOCH: {epoch}, BATCH: {batch_num} / {len(train_loader)}')
            optimizer.zero_grad()
            embeddings = self.__embedding_provider.get_embeddings(batch)

            output = model(embeddings)
            loss = loss_fn(output)
            loss.backward()
            optimizer.step()

            train_progess_monitor.step(model, epoch, batch_num, len(train_loader), loss, loss_fn)

    def __save_checkpoint(self, model, epoch):
        save_path = os.path.join(self.__checkpoint_save_path, 'isa_model_epoch_{}'.format(epoch))
        torch.save(model.state_dict(), save_path)

    def train(self, model, optimizer, temp_loss, train_ds_provider, valid_loader, epochs):
        plot_monitor = PlotMonitor()
        monitor = TrainProgressMonitor(50, valid_loader, epochs, plot_monitor)
        for epoch in range(epochs):
            train_loader = train_ds_provider()
            self.__train_epoch(model, temp_loss, optimizer, train_loader, epoch, monitor)
            self.__save_checkpoint(model, epoch)
