import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader

from rexnet.train import TrainingSpec, Recorder
from rexnet.utils import CosineLRScheduler

import tqdm

from typing import Dict, Any


class Trainer(object):
    def __init__(self, spec: TrainingSpec):
        self.spec = spec

    def train(self):
        if self.spec.distributed:
            mp.spawn(self._train, nprocs=self.spec.gpus)
        else:
            self._train(0)

    def _train(self,
               rank: int):

        if self.spec.distributed:
            torch.cuda.set_device(rank)
            dist.init_process_group(backend='nccl',
                                    init_method='tcp://127.0.0.1:8001',
                                    world_size=self.spec.gpus,
                                    rank=rank)

        self.spec.init()

        model = self.spec.construct_model().cuda()

        train_loader, valid_loader = self.spec.prepare_dataloader()

        optimizer = self.spec.create_optimizer(model.parameters())
        scheduler, epochs = self.spec.create_scheduler(optimizer)

        if self.spec.distributed:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[rank]
            )

        recorder = Recorder()

        t = tqdm.tqdm(total=epochs * len(train_loader))

        for epoch in range(epochs):
            train_metrics = self._train_step(
                rank, epoch, model, train_loader, optimizer, t)
            recorder.record(epoch, train_metrics, phase='train')

            if (epoch + 1) % self.spec.valid_epochs == 0:
                valid_metrics = self._valid_step(
                    rank, epoch, model, valid_loader, optimizer, t)
                recorder.record(epoch, valid_metrics, phase='valid')

            if rank == 0 and (epoch + 1) % self.spec.save_epochs == 0:
                ckpt = {
                    'epoch': epoch,
                    'recorder': recorder,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }
                torch.save(ckpt, self.spec.checkpoint_path)

                del ckpt

        if self.spec.distributed:
            model = model.module
        # Model save
        if rank == 0:
            torch.save(model.cpu().state_dict(), self.spec.model_save_path)

    def _train_step(self,
                    rank: int,
                    epochs: int,
                    model: nn.Module,
                    data: DataLoader,
                    optimizer: optim.Optimizer,
                    t: tqdm.tqdm) -> Dict[str, Any]:

        train_loss = 0
        train_acc = 0
        total_step = 0

        model.train()

        for idx, (pixel, label) in enumerate(data):
            pixel = pixel.cuda()
            label = label.cuda()

            optimizer.zero_grad()

            metrics = self.spec.train_objective(pixel, label, model)

            loss = metrics['loss']
            output = metrics['output']

            loss.backward()
            optimizer.step()

            t.update(1)

            # loss update
            train_loss += loss.item()

            # acc update
            _, ind = output.topk(1, 1, True, True)
            train_correct = ind.eq(label.view(-1, 1).expand_as(ind))
            train_correct_total = train_correct.view(-1).float().sum()
            train_acc += train_correct_total * (100.0 / pixel.shape[0])

            # step update
            total_step += 1

        return {'loss': train_loss / total_step, 'accuracy': train_acc / total_step}

    @torch.no_grad()
    def _valid_step(self,
                    rank: int,
                    epochs: int,
                    model: nn.Module,
                    data: DataLoader,
                    optimizer: optim.Optimizer,
                    t: tqdm.tqdm):

        valid_loss = 0
        valid_acc = 0
        total_step = 0

        model.eval()

        for idx, (pixel, label) in enumerate(data):
            pixel = pixel.cuda()
            label = label.cuda()

            metrics = self.spec.valid_objective(pixel, label, model)

            loss = metrics['loss']
            output = metrics['output']

            # loss update
            valid_loss += loss.item()

            # acc update
            _, ind = output.topk(1, 1, True, True)
            valid_correct = ind.eq(label.view(-1, 1).expand_as(ind))
            valid_correct_total = valid_correct.view(-1).float().sum()
            valid_acc += valid_correct_total * (100.0 / pixel.shape[0])

            # step update
            total_step += 1

        return {'loss': valid_loss / total_step, 'accuracy': valid_acc / total_step}
