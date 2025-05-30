import zipfile
import tarfile
import os
import logging
import requests

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

import scipy.io as sio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO)

def download_and_extract(url: str, save_dir: str, filename: Optional[str] = None) -> str:
    """
    Downloads a file from the given URL and extracts it if it is an archive.

    Args:
    - url (str): The URL of the file to download.
    - save_dir (str): Directory where the file will be saved and extracted.
    - filename (Optional[str]): Optional filename to save the file with. If None, uses the filename from the URL.

    Returns:
    - str: The full path to the directory containing the extracted files or the downloaded file.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = url.split("/")[-1]

    file_path = save_path / filename
    result_path = str(file_path)

    # Check if file already exists
    if file_path.exists():
        logging.info(f"File '{filename}' already exists in '{save_dir}'. Skipping download.")
    else:
        logging.info(f"Downloading file '{filename}' from '{url}'...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                f.write(response.content)
            logging.info(f"Download successful. File saved to: '{file_path}'")

            # Extract the file if it's an archive
            if filename.endswith(".zip"):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(save_path)
                file_path.unlink()
                result_path = str(save_path)
            elif (
                  filename.endswith(".tar.gz")
                  or filename.endswith(".tgz")
                  or filename.endswith(".gz")
                ):
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(save_path)
                file_path.unlink()
                result_path = str(save_path)
            elif filename.endswith(".tar"):
                with tarfile.open(file_path, 'r') as tar_ref:
                    tar_ref.extractall(save_path)
                file_path.unlink()
                result_path = str(save_path)
            else:
                logging.info(f"File '{filename}' is not an archive. No extraction needed.")

        except requests.RequestException as e:
            logging.error(f"Error downloading file from {url}: {str(e)}")
            raise
        except (zipfile.BadZipFile, tarfile.ReadError) as e:
            logging.error(f"Error extracting file '{filename}': {str(e)}")
            raise

    return result_path

def train_test_split(
    data: pd.DataFrame,
    test_size: Union[float, int] = 0.25,
    random_state: Union[int, None] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split arrays or matrices into random train and test subsets.

    Parameters:
    X :  pd.DataFrame
         The input data to split.
    test_size : float, int, or None, optional (default=0.25)
        If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        If None, the value is set to the complement of the train size.
    random_state : int or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If None, the random number generator is the RandomState instance used by np.random.

    Returns:
    Tuple containing:
        - data_train: pd.DataFrame
        - data_test: pd.DataFrame
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(data)
    if isinstance(test_size, float):
        test_size = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    data_train = data.iloc[train_indices]
    data_test = data.iloc[test_indices]

    return data_train, data_test


def load_labels(labels_path: str) -> pd.DataFrame:
    """
    Load labels from a MAT file.

    Args:
    - labels_path (str): Path to the labels MAT file.

    Returns:
    - pd.DataFrame: DataFrame containing the labels.
    """
    labels_mat = sio.loadmat('D:\lab3_DVC\data\labels\imagelabels.mat')
    labels_df = pd.DataFrame({"label": labels_mat["labels"][0]})

    return labels_df

def find_add_images_to_labels(images_dir: str, labels: pd.DataFrame, image_ext: str="jpg") -> pd.DataFrame:
    image_paths = sorted(
        [str(image_path.absolute()) for image_path in Path(images_dir).rglob(f"*.{image_ext}")]
    )
    if len(image_paths) != len(labels):
        logging.error(
            f"Found {len(image_paths)} image_paths but "
            f"{len(labels)} labels were provided, cannot continue."
        )
        raise ValueError

    labels_w_image_paths = labels.copy(deep=True)
    labels_w_image_paths["image_path"] = image_paths
    return labels_w_image_paths


def assign_batches(labels_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Assigns batch labels to a dataframe.

    Args:
    - labels_df (pd.Dataframe): Dataframe containing dataset information in 'label' and 'image_path' columns.
    - config (Dict[str, Any]): Configuration dictionary containing parameters for data splitting and paths.

    Returns:
    - pd.DataFrame: dataframe with assigned batch labels in 'batch_name' column.
    """
    labels_df_ = labels_df.copy(deep=True)
    labels_df_["batch_name"] = "not_set"

    n_batches = config["training"]["n_batches"]
    batch_size = len(labels_df_) // n_batches

    batch_size_current = 0
    for batch_number in range(n_batches):
        if batch_number == (n_batches -1):
            # select all the remaining data for the last batch
            labels_df_.iloc[
                    batch_size_current:,
                    labels_df_.columns.get_loc('batch_name')
            ] = str(batch_number)
        else:
            labels_df_.iloc[
                batch_size_current: batch_size_current + batch_size,
                labels_df_.columns.get_loc('batch_name')
            ] = str(batch_number)

        batch_size_current += batch_size

    return labels_df_


def select_batches(labels_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Selects data from labels_df based on 'batch_names' from config.

    Args:
    - labels_df (pd.Dataframe): .
    - config (Dict[str, Any]): Configuration dictionary containing parameters for data splitting and paths.

    Returns:
    - pd.DataFrame:
    """
    batch_names: List[str] = config["data"]["batch_names_select"]
    labels_df_ = labels_df.copy(deep=True)

    labels_df_ = labels_df_[labels_df_["batch_name"].isin(batch_names)]

    return labels_df_


def process_data(images_dir: str, labels_path: str, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Process the data by loading labels, validating data, splitting into train-validation-test sets,
    and optionally saving the splits.

    Args:
    - images_dir (str): Directory containing the images.
    - labels_path (str): Path to the labels MAT file.
    - config (Dict[str, Any]): Configuration dictionary containing parameters for data splitting and paths.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, validation, and testing DataFrames.
    """
    # Load labels
    labels_df = load_labels(labels_path)

    # Validate data (if needed, adjust based on your specific validation logic)
    valid_labels_df = find_add_images_to_labels(images_dir, labels_df)

    # Split data into train and test
    train_df, test_df = train_test_split(valid_labels_df, test_size=config.get('test_size', 0.2), random_state=config.get('random_state', 42))

    train_df = assign_batches(train_df, config)
    train_df = select_batches(train_df, config)

    # Further split train_df into train and validation
    train_df, val_df = train_test_split(train_df, test_size=config.get('val_size', 0.2), random_state=config.get('random_state', 42))

    logging.info(f"Prepared 3 data splits: train, size: {len(train_df)}, val: {len(val_df)}, test: {len(val_df)}")

    return train_df, val_df, test_df

class ImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: Optional[Any] = None) -> None:
        self.dataframe = dataframe
        self.transform = transform
        self.transform_default = transforms.Compose([
            transforms.Resize([28, 28]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        # replace with your own implementation if needed
        img_path = self.dataframe.iloc[idx]["image_path"]
        image = read_image(img_path).float() / 255.0
        # labels in this dataset start from 1, but we need to start from 0
        label = int(self.dataframe.iloc[idx]["label"]) - 1

        if self.transform:
            image = self.transform(image)
        else:
            image = self.transform_default(image)

        return image, label

def create_data_loader(
    images_dir: str,
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> DataLoader:
    """
    Create a data loader for a dataset.


    Args:
    - images_dir (str): Directory containing the images.
    - df (pd.DataFrame): DataFrame containing the dataset.
    - config (Dict[str, Any]): Configuration dictionary containing parameters for data loading.

    Returns:
    - DataLoader: DataLoader for the dataset.
    """
    transform: Optional[Any] = config.get('transform', None)
    batch_size: int = config.get('batch_size', 32)
    num_workers: int = config.get('num_workers', 2)

    dataset = ImageDataset(df, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return data_loader

def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_function: nn.Module,
        optimizer: optim.Optimizer,
        num_epochs: int,
        device: torch.device,
        save_path: Path = Path("best_model.pth")
        ) -> Path:

    model.to(device)
    best_val_loss: float = float('inf')
    best_model_path: Path = Path()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            outputs = model(batch_inputs)
            loss = loss_function(outputs, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_targets).sum().item()
            total += batch_targets.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = correct / total

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}")

        # MLflow логування метрик для тренування
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)

        # Validation
        model.eval()
        val_loss: float = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs: torch.Tensor = model(val_inputs)

                val_loss += loss_function(val_outputs, val_targets).item()
                _, val_preds = torch.max(val_outputs, 1)
                val_correct += (val_preds == val_targets).sum().item()
                val_total += val_targets.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        # MLflow логування метрик валідації
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = save_path
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Best model saved with validation loss: {best_val_loss:.4f}")

    logging.info("Training complete.")
    return best_model_path

def test_model(
        model: nn.Module,
        test_loader: DataLoader,
        loss_function: nn.Module,
        device: torch.device
        ) -> float:
    """
    Test a trained model on a test dataset and compute test metrics.


    Args:
    - model (nn.Module): Trained PyTorch model.
    - test_loader (DataLoader): DataLoader for the test dataset.
    - loss_function (nn.Module): Loss function for computing the loss.
    - device (torch.device): Device (CPU or GPU) on which to run the evaluation.

    Returns:
    - float: Test loss.
    """

    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    test_loss /= len(test_loader)
    logging.info(f"Test Loss: {test_loss:.4f}")

    # Calculate additional metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)

    return test_loss