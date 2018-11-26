import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from neurotorch.datasets.dataset import AlignedVolume, TorchVolume
import torch.cuda
import numpy as np
import pickle

class Trainer(object):
    """
    Trains a PyTorch neural network with a given input and label dataset
    """
    def __init__(self, net, inputs_volume, labels_volume, checkpoint=None,
                 optimizer=None, criterion=None, max_epochs=100000,
                 gpu_device=None, patience=None, net_filename="trained_net.bin"):
        """
        Sets up the parameters for training

        :param net: A PyTorch neural network
        :param inputs_volume: A PyTorch dataset containing inputs
        :param labels_volume: A PyTorch dataset containing corresponding labels
        """
        
        self.net_filename = net_filename

        self.max_epochs = max_epochs

        if patience is None:
            self.patience = max_epochs
        else:
            self.patience = patience

        self.best_score = None

        self.patience_count = 0

        self.best_net = None

        self.device = torch.device("cuda:{}".format(gpu_device)
                                   if gpu_device is not None
                                   else "cpu")

        self.net = net.to(self.device)

        if checkpoint is not None:
            self.net.load_state_dict(torch.load(checkpoint))

        if optimizer is None:
            self.optimizer = optim.Adam(self.net.parameters())
        else:
            self.optimizer = optimizer

        if criterion is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = criterion

        if gpu_device is not None:
            self.gpu_device = gpu_device
            self.useGpu = True

        self.volume = TorchVolume(AlignedVolume((inputs_volume,
                                                 labels_volume)))

        self.data_loader = DataLoader(self.volume,
                                      batch_size=8, shuffle=True,
                                      num_workers=4)

    def run_epoch(self, sample_batch):
        """
        Runs an epoch with a given batch of samples

        :param sample_batch: A dictionary containing inputs and labels with the keys 
"input" and "label", respectively
        """
        inputs = Variable(sample_batch[0]).float()
        labels = Variable(sample_batch[1]).float()

        inputs, labels = inputs.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()

        outputs = self.net(inputs)

        loss = self.criterion(torch.cat(outputs), labels)
        loss_hist = loss.cpu().item()
        loss.backward()
        self.optimizer.step()

        return loss_hist

    def determine_early_stop(self, loss):
        if self.best_score is None:
            self.best_score = loss
            self.best_net = self.net
            return False

        if loss < self.best_score:
            self.patience_count = 0
            self.best_score = loss
            self.best_net = self.net
            return False

        self.patience_count += 1
        return self.patience_count == self.patience

    def run_training(self):
        """
        Trains the given neural network
        """
        num_epoch = 1
        stop = False
        while num_epoch <= self.max_epochs and not stop:
            for i, sample_batch in enumerate(self.data_loader):
                if num_epoch > self.max_epochs or stop:
                    break
                loss = self.run_epoch(sample_batch)
                stop = self.determine_early_stop(loss)
                print("Epoch {}/{} ".format(num_epoch,
                                            self.max_epochs),
                      "Loss: {:.4f} ".format(loss), "Best Score: {:.4f}".format(self.best_score))
                num_epoch += 1
        with open(self.net_filename, 'wb') as f:
            pickle.dump(self.best_net, f)


class TrainerDecorator(Trainer):
    """
    A wrapper class to a features for training
    """
    def __init__(self, trainer):
        if isinstance(trainer, TrainerDecorator):
            self._trainer = trainer._trainer
        if isinstance(trainer, Trainer):
            self._trainer = trainer
        else:
            error_string = ("trainer must be a Trainer or TrainerDecorator " +
                            "instead it has type {}".format(type(trainer)))
            raise ValueError(error_string)

    def run_epoch(self, sample_batch):
        return self._trainer.run_epoch(sample_batch)

    def determine_early_stop(self, loss):
        return self._trainer.determine_early_stop(loss)

    def run_training(self):
        num_epoch = 1
        stop = False
        while num_epoch <= self._trainer.max_epochs and not stop:
            for i, sample_batch in enumerate(self._trainer.data_loader):
                if num_epoch > self._trainer.max_epochs or stop:
                    break
                loss = self.run_epoch(sample_batch)
                stop = self.determine_early_stop(loss)
                print("Epoch {}/{}".format(num_epoch,
                                           self._trainer.max_epochs),
                      "Loss: {:.4f} ".format(loss), "Best Score: {:.4f}".format(self._trainer.best_score))
                num_epoch += 1
        with open(self._trainer.net_filename, 'wb') as f:
            pickle.dump(self._trainer.best_net, f)
