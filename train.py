import argparse
import yaml
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.trainer import Trainer
from utils.helper import (
    get_dataset_class,
    get_model_class,
    get_optimizer_class,
    get_lr_scheduler,
    save_to_yaml,
)


def train(config, model_name):
    os.makedirs(config["model_dir"])
    
    dataset_class = get_dataset_class(model_name)
    model_class = get_model_class(model_name)

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

    optimizer_class = get_optimizer_class(config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"], verbose=True)

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
    save_to_yaml(config, config["model_dir"])
    
    print("Model artifacts saved to folder:", config["model_dir"])
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    parser.add_argument('--model_name', type=str, required=True, help='model type')
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    train(config, model_name=args.model_name)