import unittest
from neurotorch.core.Trainer import Trainer
from neurotorch.nets.RSUNet import RSUNet
from neurotorch.datasets.TiffDataset import TiffDataset
from neurotorch.visualization.TensorboardWriter import (LossWriter,
                                                        TrainingLogger)
import os.path
import pytest

IMAGE_PATH = "./tests/images"


class TestTrainer(unittest.TestCase):
    def test_gpu_training(self):
        net = RSUNet()
        inputs_dataset = TiffDataset(os.path.join(IMAGE_PATH,
                                                  "sample_volume.tif"))
        labels_dataset = TiffDataset(os.path.join(IMAGE_PATH,
                                                  "labels.tif"))
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=2,
                          gpu_device=0)
        trainer.run_training()

    @pytest.mark.skip()
    def test_cpu_training(self):
        net = RSUNet()
        inputs_dataset = TiffDataset(os.path.join(IMAGE_PATH,
                                                  "sample_volume.tif"))
        labels_dataset = TiffDataset(os.path.join(IMAGE_PATH,
                                                  "labels.tif"))
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=2)
        trainer.run_training()

    def test_loss_writer(self):
        net = RSUNet()
        inputs_dataset = TiffDataset(os.path.join(IMAGE_PATH,
                                                  "sample_volume.tif"))
        labels_dataset = TiffDataset(os.path.join(IMAGE_PATH,
                                                  "labels.tif"))
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=5,
                          gpu_device=0)
        trainer = LossWriter(trainer, './tests/', "test_experiment")
        trainer.run_training()

    def test_training_logger(self):
        net = RSUNet()
        inputs_dataset = TiffDataset(os.path.join(IMAGE_PATH,
                                                  "sample_volume.tif"))
        labels_dataset = TiffDataset(os.path.join(IMAGE_PATH,
                                                  "labels.tif"))
        trainer = Trainer(net, inputs_dataset, labels_dataset, max_epochs=2,
                          gpu_device=0)
        trainer = TrainingLogger(trainer, logger_dir='.')
        trainer.run_training()