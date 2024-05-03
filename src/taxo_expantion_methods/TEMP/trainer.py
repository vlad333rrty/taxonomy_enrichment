# TODO переделать потом на архитектуру с базовым тренером и моделью
import os

import torch
from tqdm import tqdm

from src.taxo_expantion_methods.common.plot_monitor import PlotMonitor, Metric
from src.taxo_expantion_methods.TEMP.temp_embeddings_provider import TEMPEmbeddingProvider


class TrainProgressMonitor:
    def __init__(self, interval: int, valid_loader, epochs: int, plot_monitor,
                 embedding_provider: TEMPEmbeddingProvider):
        self.__interval = interval
        self.__running_loss = 0
        self.__running_items = 0
        self.__valid_loader = valid_loader
        self.__epochs = epochs
        self.__plot_monitor = plot_monitor
        self.__embedding_provider = embedding_provider

    def step(self, model, epoch, i, samples, loss, loss_fn):
        self.__plot_monitor.accept(Metric('Train loss', loss.item()))
        self.__running_loss += loss.item()
        self.__running_items += 1
        if i % self.__interval == 0:
            model.eval()

            print(f'Epoch [{epoch + 1}/{self.__epochs}]. '
                  f'Batch [{i}/{samples}].'
                  f'Loss: {self.__running_loss / self.__running_items:.3f}. ')

            self.__running_loss, self.__running_items = 0.0, 0.0

            test_running_right, test_running_total = 0.0, 0.0
            for batch in self.__valid_loader:
                with torch.no_grad():
                    positive_paths = batch.positive_paths
                    negative_paths = batch.negative_paths
                    positive_paths_count = len(positive_paths)
                    embeddings = self.__embedding_provider.get_path_embeddings(positive_paths + negative_paths)
                    test_outputs = model(embeddings)
                    test_running_total += positive_paths_count + len(negative_paths)
                    test_running_right += loss_fn(positive_paths, negative_paths, test_outputs[:positive_paths_count],
                                                  test_outputs[positive_paths_count:]).sum()

            test_loss = test_running_right / test_running_total
            print(f'Test loss: {test_loss:.3f}')
            self.__plot_monitor.accept(Metric('Test loss', test_loss.cpu()))
            self.__plot_monitor.plot()

            model.train()


class TEMPTrainer:
    def __init__(self, embedding_provider: TEMPEmbeddingProvider, checkpoint_save_path):
        self.__embedding_provider = embedding_provider
        self.__checkpoint_save_path = checkpoint_save_path

    def __train_epoch(self, model, loss_fn, optimizer, train_loader, epoch,
                      train_progess_monitor: TrainProgressMonitor):
        for i, batch in (pbar := tqdm(enumerate(train_loader))):
            batch_num = i + 1
            pbar.set_description(f'EPOCH: {epoch}, BATCH: {batch_num} / {len(train_loader)}')
            optimizer.zero_grad()
            positive_paths = batch.positive_paths
            negative_paths = batch.negative_paths
            positive_paths_count = len(positive_paths)
            embeddings = self.__embedding_provider.get_path_embeddings(positive_paths + negative_paths)

            output = model(embeddings)
            loss = loss_fn(positive_paths, negative_paths, output[:positive_paths_count], output[positive_paths_count:])
            loss.backward()
            optimizer.step()

            train_progess_monitor.step(model, epoch, batch_num, len(train_loader), loss, loss_fn)

    def __save_checkpoint(self, model, epoch):
        save_path = os.path.join(self.__checkpoint_save_path, 'temp_model_epoch_{}'.format(epoch))
        torch.save(model.state_dict(), save_path)

    def train(self, model, optimizer, temp_loss, train_ds_provider, valid_loader, epochs):
        plot_monitor = PlotMonitor()
        monitor = TrainProgressMonitor(10, valid_loader, epochs, plot_monitor, self.__embedding_provider)
        for epoch in range(epochs):
            train_loader = train_ds_provider()
            self.__train_epoch(model, temp_loss, optimizer, train_loader, epoch, monitor)
            self.__save_checkpoint(model, epoch)
