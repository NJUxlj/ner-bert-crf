
import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

'''

Model Training Master Program
'''


def main(config):
    # model save path
    if os.path.exists(config['model_path']) is False:
        os.mkdir(config['model_path'])
    
    
    train_data = load_data(config['train_data_path'], config)
    
    
    model = TorchModel(Config)
    
    
    
    cuda_flag=False
    if torch.cuda.is_available():
        cuda_flag=True
        model = model.cuda()
        
    
    
    optimizer = choose_optimizer(config, model)

    
    evaluator = Evaluator(config, model, logger)

    
    
    # training procedure
    
    for epoch in range(config['epoch']):
        model.train()
        logger.info("Epoch: {}".format(epoch))
        watch_loss = []
        for batch in train_data:
            if cuda_flag:
                batch = [d.cuda() for d in batch]
            input_ids, labels = batch
            
            optimizer.zero_grad()
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
            
        print("epoch:{} loss: {}".format(epoch+1,np.mean(watch_loss)))
        # evaluator.evaluate()
        
    
    # save model weights
    model_path = os.path.join(config['model_path'], 'epoch_%d.pth'% epoch)
    # torch.save(model.state_dict(), model_path)
    
    return model, train_data


if __name__ == '__main__':
    from config import Config
    main(Config)