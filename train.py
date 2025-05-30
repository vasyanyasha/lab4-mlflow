import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import yaml

from download_data import download_and_prepare_data

from model import ResNet18
from utils import create_data_loader, train_model, device

def train_pipeline(config, images_dir, train_df, val_df):
    train_loader = create_data_loader(images_dir, train_df, config)
    val_loader = create_data_loader(images_dir, val_df, config)

    model = ResNet18(in_channels=3, num_classes=102).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["training"]["lr"])

    best_model_path = train_model(
        model, train_loader, val_loader, loss_function, optimizer,
        num_epochs=config["training"]["num_epochs"],
        device=device
    )

    return model, best_model_path

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    images_dir, train_df, val_df, _ = download_and_prepare_data(config)
    model, _ = train_pipeline(config, images_dir, train_df, val_df)

    # Save the trained model
    os.makedirs("model", exist_ok=True)
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)