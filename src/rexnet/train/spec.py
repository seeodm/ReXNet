import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from rexnet.data import AffectNetDataset
from rexnet.utils import CosineLRScheduler

from typing import Tuple, Dict, Iterator, Any


class TrainingSpec(object):
    def init(self):
        pass

    def prepare_transform(self) -> Tuple[transforms.Compose, transforms.Compose]:
        raise NotImplementedError()

    def prepare_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        raise NotImplementedError()

    def construct_model(self) -> nn.Module:
        raise NotImplementedError()

    def create_optimizer(self, params: Iterator[nn.Parameter]
                         ) -> optim.Optimizer:
        raise NotImplementedError()

    def create_scheduler(self, optimizer: optim.Optimizer) -> Tuple[CosineLRScheduler, int]:
        raise NotImplementedError()

    def train_objective(self, pixel: torch.Tensor, label: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
        raise NotImplementedError()

    def valid_objective(self, pixel: torch.Tensor, label: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
        raise NotImplementedError()
