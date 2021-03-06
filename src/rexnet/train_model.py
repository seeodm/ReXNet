import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from rexnet.data import AffectNetDataset
from rexnet.model import ReXNetV1
from rexnet.train import TrainingSpec, Trainer
from rexnet.utils import CosineLRScheduler
from rexnet.loss import compute_center_loss, get_center_delta

from typing import Tuple, Dict, Iterator, Any


class RexnetTrainingSpec(TrainingSpec):
    def __init__(self, train_data: str, valid_data: str, train_batch_size: int,
                 train_shuffle: bool, valid_batch_size: int, valid_shuffle: bool, num_workers: int,
                 base_lr: float, lr_min: float, lr_decay: float, warmup_lr_init: float,
                 warmup_t: int, cooldown_epochs: int, momentum: float, nesterov: bool,
                 epochs: int, save_epochs: int, valid_epochs: int,
                 model_save_path: str, checkpoint_path: str,
                 gpus: int, center_loss: bool, center_loss_lambda: float, center_loss_alpha: float):
        self.train_data = train_data
        self.valid_data = valid_data

        self.train_batch_size = train_batch_size
        self.train_shuffle = train_shuffle
        self.valid_batch_size = valid_batch_size
        self.valid_shuffle = valid_shuffle
        self.num_workers = num_workers

        self.base_lr = base_lr
        self.lr_min = lr_min
        self.lr_decay = lr_decay

        self.warmup_lr_init = warmup_lr_init
        self.warmup_t = warmup_t

        self.cooldown_epochs = cooldown_epochs

        self.momentum = momentum
        self.nesterov = nesterov

        self.epochs = epochs
        self.save_epochs = save_epochs
        self.valid_epochs = valid_epochs

        self.model_save_path = model_save_path
        self.checkpoint_path = checkpoint_path

        self.gpus = gpus
        self.distributed = True if self.gpus is not None else False

        self.center_loss = center_loss
        self.center_loss_lambda = center_loss_lambda
        self.center_loss_alpha = center_loss_alpha

    def init(self):
        self.criterion = nn.CrossEntropyLoss()

    def construct_model(self):
        return ReXNetV1(center_loss=self.center_loss)

    def prepare_transform(self) -> Tuple[transforms.Compose, transforms.Compose]:
        train_transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop((224, 224), scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        valid_transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
        ])

        return train_transformer, valid_transformer

    def prepare_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        train_transformer, valid_transformer = self.prepare_transform()

        train_dataset = AffectNetDataset(path=self.train_data,
                                         transform=train_transformer,
                                         phase='train')
        valid_dataset = AffectNetDataset(path=self.valid_data,
                                         transform=valid_transformer,
                                         phase='validation')

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.train_batch_size,
                                  shuffle=self.train_shuffle,
                                  num_workers=self.num_workers)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=self.valid_batch_size,
                                  shuffle=self.valid_shuffle,
                                  num_workers=self.num_workers)

        return train_loader, valid_loader

    def create_optimizer(self, params: Iterator[nn.Parameter]
                         ) -> optim.Optimizer:
        optimizer = optim.SGD(params,
                              lr=self.base_lr,
                              momentum=self.momentum,
                              nesterov=self.nesterov)

        return optimizer

    def create_scheduler(self, optimizer: optim.Optimizer) -> Tuple[CosineLRScheduler, int]:
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=self.epochs,
            t_mul=1,
            lr_min=self.lr_min,
            decay_rate=self.lr_decay,
            warmup_lr_init=self.warmup_lr_init,
            warmup_t=self.warmup_t,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct=0.67,
            noise_std=1,
            noise_seed=42,
        )

        total_epochs = scheduler.get_cycle_length() + self.cooldown_epochs

        return scheduler, total_epochs

    def train_objective(self, pixel: torch.Tensor, label: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
        if self.center_loss:
            output, feature = model(pixel)
            if self.distributed:
                centers = model.module.centers
            else:
                centers = model.centers

            loss_nll = self.criterion(output, label)
            loss_center = compute_center_loss(feature, centers, label)

            loss = loss_nll + self.center_loss_lambda * loss_center

            center_delta = get_center_delta(
                feature, centers, label, alpha=self.center_loss_alpha)

            if self.distributed:
                model.module.centers.data -= center_delta
            else:
                model.centers.data -= center_delta
        else:
            output = model(pixel)

            loss = self.criterion(output, label)

        return {'output': output, 'loss': loss}

    def valid_objective(self, pixel: torch.Tensor, label: torch.Tensor, model: nn.Module) -> Dict[str, Any]:
        if self.center_loss:
            output, feature = model(pixel)
            if self.distributed:
                centers = model.module.centers
            else:
                centers = model.centers

            loss_nll = self.criterion(output, label)
            loss_center = compute_center_loss(feature, centers, label)

            loss = loss_nll + self.center_loss_lambda * loss_center
        else:
            output = model(pixel)

            loss = self.criterion(output, label)

        return {'output': output, 'loss': loss}


def train_rexnet(args: argparse.Namespace):
    spec = RexnetTrainingSpec(train_data=args.train_data, valid_data=args.valid_data,
                              train_batch_size=args.train_batch_size, train_shuffle=args.train_shuffle,
                              valid_batch_size=args.valid_batch_size, valid_shuffle=args.valid_shuffle,
                              num_workers=args.num_workers, base_lr=args.base_lr, lr_min=args.lr_min, lr_decay=args.lr_decay,
                              warmup_lr_init=args.warmup_lr_init, warmup_t=args.warmup_t, cooldown_epochs=args.cooldown_epochs,
                              momentum=args.momentum, nesterov=args.nesterov,
                              epochs=args.epochs, save_epochs=args.save_epochs, valid_epochs=args.valid_epochs,
                              model_save_path=args.model_save_path, checkpoint_path=args.checkpoint_path,
                              gpus=args.gpus, center_loss=args.center_loss, center_loss_lambda=args.center_loss_lambda,
                              center_loss_alpha=args.center_loss_alpha)

    Trainer(spec).train()


def add_subparser(subparsers):
    parser = subparsers.add_parser('train', help='train ReXNet')

    group = parser.add_argument_group('Dataset')
    group.add_argument('--train_data', required=True,
                       help='affectnet train data file path')
    group.add_argument('--valid_data', required=True,
                       help='affectnet valid data file path')

    group = parser.add_argument_group('Dataset Config')
    group.add_argument('--train_batch_size', default=256,
                       type=int, help='train batch size')
    group.add_argument('--valid_batch_size', default=256,
                       type=int, help='valid batch size')
    group.add_argument('--train_shuffle', default=True,
                       type=bool, help='train data shuffle')
    group.add_argument('--valid_shuffle', default=True,
                       type=bool, help='valid data shuffle')
    group.add_argument('--num_workers', default=4, type=int,
                       help='number of workers for data load')

    group = parser.add_argument_group('Train Config')
    group.add_argument('--epochs', default=400, type=int,
                       help='num of total epochs')
    group.add_argument('--save_epochs', default=10, type=int,
                       help='interval epohcs of saving')
    group.add_argument('--valid_epochs', default=1, type=int,
                       help='interval epochs of valid')
    group.add_argument('--base_lr', default=0.5,
                       type=float, help='base lr value')
    group.add_argument('--lr_min', default=0.00001,
                       type=float, help='minimum value of lr')
    group.add_argument('--lr_decay', default=0.1,
                       type=float, help='lr decay value')
    group.add_argument('--warmup_lr_init', default=0.0001,
                       type=float, help='base warmup lr')
    group.add_argument('--warmup_t', default=3, type=int, help='warmup epochs')
    group.add_argument('--cooldown_epochs', default=10,
                       type=int, help='cooldown epochs')

    group = parser.add_argument_group('Optimizer Config')
    group.add_argument('--momentum', default=0.9,
                       type=float, help='momentum of SGD')
    group.add_argument('--nesterov', default=True, type=bool)

    group = parser.add_argument_group('Saving Config')
    group.add_argument('--model_save_path',
                       default='./model.pth', help='model save path')
    group.add_argument('--checkpoint_path',
                       default='./checkpoint.pth', help='checkpoint save path')

    group = parser.add_argument_group('Others')
    group.add_argument('--gpus', default=None, type=int,
                       help='number of gpu devices')
    group.add_argument('--center_loss', default=True,
                       type=bool, help='use center loss')
    group.add_argument('--center_loss_lambda', default=0.01,
                       type=float, help='center loss lambda')
    group.add_argument('--center_loss_alpha', default=0.5,
                       type=float, help='center loss alpha')

    parser.set_defaults(func=train_rexnet)
