import torch
from torch import nn
from torch.nn import functional as F

from abc import ABC, abstractmethod

from steps.utils import get_batches
from steps.utils import evaluate_loss
from steps.models.FeedForward import SimpleModel
from steps.models.llama import RopeModel

import time
import pandas as pd
from matplotlib import pyplot as plt

class Model(ABC):
    '''
    Abstract class for all models
    '''

    @abstractmethod
    def train(self):
        pass




class llama(Model):

    def __init__(self,config):
        self.config = config
        self.model = RopeModel(self.config)

    def train(self,dataset, scheduler=None, print_logs=False):

        optimizer = torch.optim.Adam(
            self.model.parameters(), 
        )

        losses = []
        start_time = time.time()
        for epoch in range(self.config['epochs']):
            optimizer.zero_grad()

            xs, ys = get_batches(dataset, 'train', self.config['batch_size'], self.config['context_window'])
            logits, loss = self.model(xs, targets=ys)
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            if epoch % self.config['log_interval'] == 0:
                batch_time = time.time() - start_time
                x = evaluate_loss(self.model,dataset,self.config)
                losses += [x]
                if print_logs:
                    print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (self.config['epochs'] - epoch)/self.config['log_interval'] :.3f}")
                start_time = time.time()

                if scheduler:
                    print("lr: ", scheduler.get_lr())

        print("validation loss: ", losses[-1]['val'])
        return self.model, pd.DataFrame(losses).plot()
