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

        if rank == 0:
            t = tqdm.tqdm(total=epochs * len(train_loader))

        for epoch in range(epochs):
            train_metrics = self._train_step(epoch, model, train_loader, optimizer, t)
            recorder.record(epoch, train_metrics, phase='train')

            if (epoch + 1) % self.spec.eval_epochs:
                eval_metrics = self._valid_step(epoch, model, valid_loader, optimizer, t)
                recorder.record(epoch, eval_metrics, phase='eval')

            if rank == 0 and (epoch + 1) % self.spec.save_epochs:
                ckpt = {
                    'epoch' : epoch,
                    'recorder' : recorder,
                    'model' : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                }
                torch.save(ckpt, self.spec.checkpoint_path)

                del ckpt

        # Model save
        if rank == 0:
            torch.save(model.cpu().state_dict(), self.spec.model_save_path)

    def _train_step(self, 
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

        return {'loss' : train_loss / total_step, 'accuracy' : train_acc / total_step}

    @torch.no_grad()
    def _valid_step(self,
                    epochs: int,
                    model: nn.Module,
                    data: DataLoader,
                    optimizer: optim.Optimizer,
                    t: tqdm.tqdm):

        eval_loss = 0
        eval_acc = 0
        total_step = 0

        model.eval()

        for idx, (pixel, label) in enumerate(data):
            pixel = pixel.cuda()
            label = label.cuda()

            metrics = self.spec.valid_objective(pixel, label, model)

            loss = metrics['loss']
            output = metrics['output']

            # loss update
            eval_loss += loss.item()

            # acc update
            _, ind = output.topk(1, 1, True, True)
            eval_correct = ind.eq(label.view(-1, 1).expand_as(ind))
            eval_correct_total = eval_correct.view(-1).float().sum()
            eval_acc += eval_correct_total * (100.0 / pixel.shape[0])

            # step update
            total_step += 1

        return {'loss' : eval_loss / total_step, 'accuracy' : eval_acc / total_step}

