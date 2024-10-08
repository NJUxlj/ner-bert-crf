import torch
import re
import numpy as np
from collections import defaultdict

from loader import load_data



class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    
    def eval(self):
        pass
    
    
    
    
    
    def write_states(self):
        pass
    
    
    
    def show_states(self):
        pass
    
    
    
    
    
    def decode(self):
        pass