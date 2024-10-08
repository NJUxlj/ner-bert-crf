import re
import json
import torch
import os
import jieba
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader    


class DataGenerator(Dataset):
    
    def __init__(self, config, data_path):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        
        
        
        self.load()
    
    
    def load(self):
        self.data = []
    
    
    
    def encode_sentence(self, sentence):
        pass
    
    
    def padding(self, input_id, pad_token=0):
        pass
    
    
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self,index):
        return self.data[index]
    
    
    


def load_vocab(path):
    return json.load(open(path))
    
    
        