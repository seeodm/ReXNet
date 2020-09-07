import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from rexnet.train import TrainingSpec
from rexnet.utils import CosineLRScheduler

import tqdm

class Trainier(object):
    def __init__(self, spec: TrainingSpec):
        self.spec = spec

    def train(self):
        self.spec.init()

        model = self.spec.construct_model().cuda()

        train_loader, valid_loader = self.spec.prepare_dataloader()

        optimizer = self.spec.create_optimizer(model.parameters())
        scheduler, epochs = self.spec.create_scheduler(optimizer)

        for epoch in tqdm.tqdm(range(epochs)):
            self._train_step(epoch, model, train_loader, optimizer)
            self._valid_step(epoch, model, valid_loader, optimizer)

        torch.save(model.cpu().state_dict(), self.spec.model_save_path)

    def _train_step(self, 
                    epochs: int,
                    model: nn.Module,
                    data: DataLoader,
                    optimizer: optim.Optimizer):
        
        model.train()

        for idx, (pixel, label) in enumerate(data):
            pixel = pixel.cuda()
            label = label.cuda()

            optimizer.zero_grad()

            loss = self.spec.train_objective(pixel, label, model)

            loss.backward()
            optimizer.step()

    @torch.no_grad()
    def _valid_step(self,
                    epochs: int,
                    model: nn.Module,
                    data: DataLoader,
                    optimizer: optim.Optimizer):
        model.eval()

        for idx, (pixel, label) in enumerate(data):
            pixel = pixel.cuda()
            label = label.cuda()

            loss = self.spec.valid_objective(pixel, label, model)

        

