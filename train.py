import argparse
import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils.model import CBOW_Model, SkipGram_Model
from utils.dataset import CBOW_Dataset, SkipGram_Dataset
from utils.trainer import Trainer


def train(config, model_name):
    
    
    if model_name == 'cbow':
        dataset_class = CBOW_Dataset
        model_class = CBOW_Model
    elif model_name == 'skipgram':
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
        train_dataset, batch_size=config["train_batch_size"], shuffle=True, drop_last=True
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
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        factor=config["lr_scheduler_factor"],
        patience=config["lr_scheduler_patience"],
        threshold=0.01,
        verbose=True,
    )
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
        early_stopping_wait=config['early_stopping_wait'],
        device=device,
        model_dir=config["model_dir"],
        model_name=model_name,
    )

    trainer.train()
    print("Training finished.")

    model_path = os.path.join(config["model_dir"], f"{model_name}_model.pt")
    trainer.save_model(model_path)
    print("Model saved to:", model_path)
    
    vocab_path = os.path.join(config["model_dir"], "vocab.pt")
    torch.save(train_dataset.vocab, vocab_path)
    print("Vocabulary saved to:", vocab_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    parser.add_argument('--model_name', type=str, required=True, help='model type')
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    train(config, model_name=args.model_name)