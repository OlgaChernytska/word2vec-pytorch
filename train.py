import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.model import CBOW_Model
from utils.dataset import CBOW_Dataset
from utils.trainer import Trainer


def train(config):
    train_dataset = CBOW_Dataset(
        name=config["dataset"],
        set_type="train",
        data_dir=config["data_dir"],
        vocab=None,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=500, shuffle=True, drop_last=True
    )

    val_dataset = CBOW_Dataset(
        name=config["dataset"],
        set_type="valid",
        data_dir=config["data_dir"],
        vocab=train_dataset.vocab,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=500, shuffle=True, drop_last=True
    )

    print("Train Dataset Size:", train_dataset.__len__())
    print("Val Dataset Size:", val_dataset.__len__())

    vocab_size = train_dataset.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    model = CBOW_Model(vocab_size=vocab_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
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
        device=device,
    )

    trainer.train()
    print("Training finished.")

    model_path = os.path.join(config["model_dir"], "model.pt")
    trainer.save_model(model_path)
    print("Model saved to:", model_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    train(config)