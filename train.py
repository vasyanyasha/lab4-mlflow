import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import os
import yaml

from download_data import download_and_prepare_data

from model import ResNet18
from utils import create_data_loader, train_model, device

import mlflow

def train_pipeline(config, images_dir, train_df, val_df):
    # логування параметрів
    mlflow.log_params({
        "learning_rate": config["training"]["lr"],
        "batch_size": config["dataloader"]["batch_size"],
        "num_epochs": config["training"]["num_epochs"],
        "seed": config["data"]["random_state"]
    })

    # Далі все як раніше
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

    # логування артефактів моделі
    mlflow.pytorch.log_model(model, "model")

    return model, best_model_path