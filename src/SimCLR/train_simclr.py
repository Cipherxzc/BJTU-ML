import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from .simclr import SimCLR
from ..Dataset.dataset import SimCLRDataset
from ..Dataset.transform import Augmentation

def train_simclr(train_data_path, model_save_path, num_epochs=10, batch_size=256):
    train_data = pd.read_csv(train_data_path)

    print(f"Train data shape: {train_data.shape}")

    train_dataset = SimCLRDataset(train_data, Augmentation())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimCLR(out_dim=128).to(device)
    model.fit(train_loader, epochs=num_epochs, device=device)

    save_file(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SimCLR model.")
    parser.add_argument("--train-data", required=True, help="Path to the training data CSV file.")
    parser.add_argument("--test-data", required=False, help="Path to the testing data CSV file.")
    parser.add_argument("--model-save-path", required=True, help="Path to save the trained model.")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training.")
    args = parser.parse_args()

    train_simclr(args.train_data, args.model_save_path, args.num_epochs, args.batch_size)