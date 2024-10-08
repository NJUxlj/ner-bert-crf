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
        self.config['vocab_size'] = len(self.vocab)
        
        self.sentences = [] # 用来存储数据集中的所有句子
        
        # 加载NER任务的专属vocab
        self.schema = self.load_schema(config['schema_path'])
        
        
        self.load()
    
    
    def load(self):
        self.data = []
        
        with open(self.path, encoding='utf8') as f:
            batches = f.read().split("\n\n")
            
            for batch in batches:
                sentence = [] # 收集每个batch中的所有token
                labels = []
                
                for example in batch.split("\n"):
                    if example.strip() == "":
                        continue
                    char, label = example.split() # 分离 x,y
                    sentence.append(char)
                    labels.append(self.schema[label])
                
                self.sentences.append("".join(sentence))
                input_ids = self.encode_sentence(sentence) # 默认padding, pad_token=0
                labels = self.padding(labels, -1)  # pad_token = -1
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
                
        return

    
    
    
    def encode_sentence(self, text, padding = True):
        input_id = []
        
        for word in text:
            input_id.append(self.vocab.get(word,self.vocab['[UNK]']))
        
        if padding:
            return self.padding(input_id)
        else:
            return input_id
    
    
    def padding(self, input_id, pad_token=0):
        # padding
        input_id += [pad_token]*(self.config['max_length']-len(input_id))
        
        # truncate
        return input_id[:self.config['max_length']]
    
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self,index):
        return self.data[index]
    
    
    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)
    
    
    


def load_vocab(path):
    vocab_dict = {}
    
    with open(path, encoding="utf8") as f:
        for index, line in enumerate(f):
            vocab_dict[line.strip()] = index+1 # 0 is reserved for padding
    return vocab_dict
    
        
def load_data(data_path, config, shuffle=True):
    '''
     use Pytorch DataLoader to encapsulate dataset
    '''
    
    data_generator = DataGenerator(config, data_path)
    
    data_loader = DataLoader(data_generator, batch_size=config['batch_size'], shuffle=shuffle)

    return data_loader




# test 
if __name__ == "__main__":
    from config import Config
    dg = DataGenerator(Config, Config['train_data_path'])