# TODO переделать потом на архитектуру с базовым тренером и моделью
import os
import time

import torch
from transformers import BertTokenizer, BertModel

from src.taxo_expantion_methods.TEMP.plot_monitor import PlotMonitor, Metric
from src.taxo_expantion_methods.TEMP.temp_embeddings_provider import TEMPEmbeddingProvider
from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.datasets_processing.temp_dataset_generator import TEMPDsCreator
from src.taxo_expantion_methods.utils.utils import get_synset_simple_name


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

    def step(self, model, epoch, i, samples, loss, loss_fn, device):
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
                positive_paths = batch.positive_paths
                negative_paths = batch.negative_paths
                positive_paths_count = len(positive_paths)
                embeddings = torch.stack(
                    list(map(self.__embedding_provider.get_path_embedding, positive_paths + negative_paths)))
                test_outputs = model(embeddings.to(device))
                test_running_total += len(embeddings)
                test_running_right += loss_fn(positive_paths, negative_paths, test_outputs[:positive_paths_count],
                                              test_outputs[positive_paths_count:]).sum()

            test_loss = test_running_right / test_running_total
            print(f'Test loss: {test_loss:.3f}')
            self.__plot_monitor.plot()

            model.train()


class TEMPTrainer:
    def __init__(self, embedding_provider: TEMPEmbeddingProvider, checkpoint_save_path):
        self.__embedding_provider = embedding_provider
        self.__checkpoint_save_path = checkpoint_save_path

    def __train_epoch(self, model, loss_fn, optimizer, train_loader, epoch, device,
                      train_progess_monitor: TrainProgressMonitor):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            positive_paths = batch.positive_paths
            negative_paths = batch.negative_paths
            positive_paths_count = len(positive_paths)
            start = time.time()
            embeddings = torch.stack(
                list(map(self.__embedding_provider.get_path_embedding, positive_paths + negative_paths)))
            print('Got embeddings for {} positive and {} negative samples in {}s'.format(positive_paths_count,
                  len(negative_paths), time.time() - start))

            output = model(embeddings)
            loss = loss_fn(positive_paths, negative_paths, output[:positive_paths_count], output[positive_paths_count:])
            loss.backward()
            optimizer.step()

            train_progess_monitor.step(model, epoch, i, len(train_loader), loss, loss_fn, device)

    def __save_checkpoint(self, model, optimizer, epoch):
        save_path = os.path.join(self.__checkpoint_save_path, 'temp_model_epoch_{}'.format(epoch))
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, save_path)

    def train(self, model, optimizer, temp_loss, train_ds_provider, valid_loader, epochs, device):
        plot_monitor = PlotMonitor()
        monitor = TrainProgressMonitor(1, valid_loader, epochs, plot_monitor, self.__embedding_provider)
        for epoch in range(epochs):
            train_loader = train_ds_provider()
            self.__train_epoch(model, temp_loss, optimizer, train_loader, epoch, device, monitor)
            self.__save_checkpoint(model, optimizer, epoch)
