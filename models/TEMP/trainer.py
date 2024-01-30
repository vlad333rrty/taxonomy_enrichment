# TODO переделать потом на архитектуру с базовым тренером и моделью
class TrainProgressMonitor:
    def __init__(self, interval: int, valid_loader, epochs: int):
        self.__interval = interval
        self.__running_loss = 0
        self.__running_items = 0
        self.__valid_loader = valid_loader
        self.__epochs = epochs

    def step(self, model, epoch, i, loss, device):
        self.__running_loss += loss.item()
        self.__running_items += 1
        if i % self.__interval == 0:
            model.eval()

            print(f'Epoch [{epoch + 1}/{self.__epochs}]. '
                  f'Loss: {self.__running_loss / self.__running_items:.3f}. ')

            self.__running_loss, self.__running_items, running_right = 0.0, 0.0, 0.0

            test_running_right, test_running_total = 0.0, 0.0
            for i, data in enumerate(self.__valid_loader):
                paths, batch = data[0], data[1]
                test_outputs = model(batch.to(device))
                test_running_total += len(batch)
                test_running_right += loss(batch.to(device), test_outputs).sum()

            print(f'Test loss: {test_running_right / test_running_total:.3f}')

            model.train()


def __train_epoch(model, loss_fn, optimizer, train_loader, epoch, device, train_progess_monitor: TrainProgressMonitor):
    for i, mini_batch in enumerate(train_loader):
        optimizer.zero_grad()

        paths, batch = mini_batch[0], mini_batch[1]
        output = model(batch)
        loss = loss_fn(output[0], output[1:], paths)
        loss.backward()
        optimizer.step()

        train_progess_monitor.step(model, epoch, i, loss, device)


def train(model, loss_fn, optimizer, train_loader, valid_loader, epochs, device):
    monitor = TrainProgressMonitor(500, valid_loader, epochs)
    for epoch in epochs:
        __train_epoch(model, loss_fn, optimizer, train_loader, epoch, device, monitor)
