import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF




class TorchModel(nn.Module):
  def _init__(self, config):
    super().__init__()

  def forward(self, x, target = None):
    pass






def choose_optimizer(config, model):
  pass




if __name__ == '__main__':
    from config import Config
    model = TorchModel(Config)
  
  
