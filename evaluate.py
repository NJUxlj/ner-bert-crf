import torch
import re
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from loader import load_data



class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data: DataLoader = load_data(config["valid_data_path"], config, shuffle=False)

    
    def eval(self, epoch):
        self.logger.info(f"开始测试第{epoch}轮模型效果")
        self.stats_dict = {
            "LOCATION": defaultdict(int),
            "TIME": defaultdict(int),
            "PERSON":defaultdict(int),
            "ORGANIZATION":defaultdict(int),
        }
        
        
        for index, batch in enumerate(self.valid_data):
            # get a batch of sentences
            sentences = self.valid_data.dataset.sentences[index*self.config['batch_size']:(index+1)*self.config['batch_size']]
                
            if torch.cuda.is_available():
                batch = [d.cuda() for d in batch]
            
            input_ids_list, labels_list = batch # 1 tokens -> 1 label
            # input_ids_list: [batch_size, max_length]
            # labels_list: [batch_size, max_length]
            with torch.no_grad():
                # predict
                predicts = self.model(input_ids_list)
            self.write_states(labels_list, predicts, sentences)
        self.show_states()
        return
            
            
    
    
    
    
    
    def write_states(self, labels, predicts, sentences):
        '''
            处理一个 batch的数据，并写入统计字典，统计预测结果与真实结果的差异
        '''
        assert len(labels)==len(predicts)==len(sentences)
        
        '''
            .cpu().detach().tolist() 的含义:
                .cpu()：将数据从 GPU 转移回 CPU，以便脱离计算图后再处理。
                .detach()：阻止梯度传播，避免影响后续的反向传播。
                .tolist()：将张量转换为普通的 Python 列表。
        '''
        if not self.config['use_crf']:
            pass
    
    
    def show_states(self):
        pass
    
    
    
    
    
    def decode(self):
        pass
    
    def write_stats_to_csv():
        pass