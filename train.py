import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from utils.model import CBOW_Model, SkipGram_Model
from utils.dataset import CBOW_Dataset, SkipGram_Dataset
from utils.trainer import Trainer


def train(config, model_name):

    if model_name == "cbow":
        dataset_class = CBOW_Dataset
        model_class = CBOW_Model
    elif model_name == "skipgram":
        dataset_class = SkipGram_Dataset
        model_class = SkipGram_Model
    else:
        raise ValueError("Choose model_name from: cbow, skipgram")
        return

    train_dataset = dataset_class(
        name=config["dataset"],
        set_type="train",
        data_dir=config["data_dir"],
        vocab=None,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        drop_last=True,
    )

    val_dataset = dataset_class(
        name=config["dataset"],
        set_type="valid",
        data_dir=config["data_dir"],
        vocab=train_dataset.vocab,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["val_batch_size"], shuffle=True, drop_last=True
    )
    print("Train Dataset Size:", train_dataset.__len__())
    print("Val Dataset Size:", val_dataset.__len__())

    vocab_size = train_dataset.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    model = model_class(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    lr_lambda = lambda epoch: (config["epochs"] - epoch) / config["epochs"]
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=config["val_steps"],
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=config["checkpoint_frequency"],
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["model_dir"],
        model_name=model_name,
    )

    trainer.train()
    print("Training finished.")
    
    trainer.save_model()
    trainer.save_loss()
    trainer.save_vocab()
    print("Model saved to folder:", config["model_dir"])
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    parser.add_argument('--model_name', type=str, required=True, help='model type')
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    train(config, model_name=args.model_name)