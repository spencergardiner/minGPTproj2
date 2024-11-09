import sys

sys.path.append('../..')

from mingpt.model import GPT
from mingpt.training_data import NaiveJsonlDataset
from mingpt.trainer import Trainer
from mingpt.utils import CfgNode as CN

from torch.utils.data import DataLoader


def model_config():
    C = GPT.get_default_config()
    C.vocab_size = 50257  # pulled from HF docs, default # tokens for GPT2
    C.block_size = 2048 # context window size. Made it bigger than defualt just for funzies
    C.model_type = 'gpt-mini'
    
    return C

def training_config():
        C = CN()
        # device to train on
        C.device = 'cuda'
        # dataloder parameters
        C.num_workers = 4
        # optimizer parameters
        C.max_iters = 7E6
        C.batch_size = 1
        C.learning_rate = 5e-5
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C
    

if __name__ == "__main__":
    print(model_config())
    print(training_config())
    
    dataset = NaiveJsonlDataset()
    

    model = GPT(model_config())
    
    trainer = Trainer(training_config(), model, dataset)
    
    trainer.run()  
    