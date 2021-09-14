import os
import yaml
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from utils.model import CBOW_Model, SkipGram_Model
from utils.dataset import CBOW_Dataset, SkipGram_Dataset


def get_dataset_class(model_name: str):
    if model_name == "cbow":
        return CBOW_Dataset
    elif model_name == "skipgram":
        return SkipGram_Dataset
    else:
        raise ValueError("Choose model_name from: cbow, skipgram")
        return


def get_model_class(model_name: str):
    if model_name == "cbow":
        return CBOW_Model
    elif model_name == "skipgram":
        return SkipGram_Model
    else:
        raise ValueError("Choose model_name from: cbow, skipgram")
        return


def get_optimizer_class(name: str):
    if name == "Adam":
        return optim.Adam
    elif name == "SGD":
        return optim.SGD
    else:
        raise ValueError("Choose optimizer from: Adam, SGD")
        return
    

def get_lr_scheduler(optimizer, total_epochs: int, verbose: bool = True):
    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
    return lr_scheduler


def save_to_yaml(config: dict, model_dir: str):
    config_path = os.path.join(model_dir, "config.yaml")
    with open(config_path, "w") as stream:
        yaml.dump(config, stream)
    