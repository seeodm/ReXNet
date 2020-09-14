import torch

from typing import Dict, Tuple


class Recorder(object):
    def __init__(self):
        self.train_metrics = {
            'loss': [],
            'accuracy': [],
        }
        self.valid_metrics = {
            'loss': [],
            'accuracy': [],
        }

    def record(self, epoch, metrics, phase='train'):
        if phase == 'train':
            self.train_metrics['loss'].append((epoch, metrics['loss']))
            self.train_metrics['accuracy'].append((epoch, metrics['accuracy']))
        else:
            self.valid_metrics['loss'].append((epoch, metrics['loss']))
            self.valid_metrics['accuracy'].append((epoch, metrics['accuracy']))
