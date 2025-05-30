from utils import create_data_loader, test_model, device
import torch.nn as nn

def evaluate_model(model, images_dir, test_df, config):
    test_loader = create_data_loader(images_dir, test_df, config)
    test_model(model, test_loader, nn.CrossEntropyLoss(), device=device)