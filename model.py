import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF




class TorchModel(nn.Module):
	def __init__(self, config):	
		super(TorchModel,self).__init__()
		hidden_size = config["hidden_size"]
		# 必须先跑loader，获得vocab_size
		vocab_size = config["vocab_size"] + 1  # leave 0 for padding
		max_length = config["max_length"]
		class_num = config["class_num"]
		num_layers = config["num_layers"]
		self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
		self.bilstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
		self.classify = nn.Linear(hidden_size * 2, class_num) 
		# crf层, 用来计算 emission score tensor
		self.crf_layer = CRF(class_num, batch_first=True)
		self.use_crf = config["use_crf"]
		# -1 is the padding value for labels, which will be ignored in loss calculation
		self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

	def forward(self, x, target = None):
		x = self.embedding(x) # (batch_size, seq_len)
  
		x,_ = self.bilstm(x) # (batch_size, seq_len, hidden_size * 2)

		predict = self.classify(x) # (batch_size, seq_len, class_num)
		
		if target is not None:
			if self.use_crf:
				mask = target.gr(-1)
				# crf自带cross entropy loss
                # CRF loss 最后需要取反
				return - self.crf_layer(predict, target, mask, reduction = 'mean')
         		
			else:
				return self.loss(predict.view(-1, self.class_num), target.view(-1))
		else:
			if self.use_crf:
				# 维特比解码
				return self.crf_layer.decode(predict) # (batch_size, seq_len)
			else:
				return predict




def choose_optimizer(config, model):
	optimizer = config['optimizer']
	learning_rate = config['learning_rate']
 
	if optimizer == 'adam':
		return Adam(model.parameters(), lr=learning_rate)
	elif optimizer == 'sgd':
		return SGD(model.parameters(), lr=learning_rate)




if __name__ == '__main__':
	from config import Config
	model = TorchModel(Config)

  
