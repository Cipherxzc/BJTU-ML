import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
from safetensors.torch import save_file
from tqdm import tqdm
from ..Dataset.dataset import FashionMNISTDataset

def train_resnet(train_data_path, test_data_path, model_save_path, num_epochs=10, batch_size=256, learning_rate=0.001):
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    transform = transforms.Compose([
        transforms.Resize(224),  # 调整图像大小为 224x224
        transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为 RGB 图像
        transforms.ToTensor(),  # 将图像转换为 torch.Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    train_dataset = FashionMNISTDataset(train_data, transform=transform)
    test_dataset = FashionMNISTDataset(test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, train_dataset.num_classes)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=running_loss/len(train_loader))
                pbar.update(1)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.update(1)

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

    save_file(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet model.")
    parser.add_argument("--train-data", required=True, help="Path to the training data CSV file.")
    parser.add_argument("--test-data", required=True, help="Path to the testing data CSV file.")
    parser.add_argument("--model-save-path", required=True, help="Path to save the trained model.")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs for training.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for training.")
    args = parser.parse_args()

    train_resnet(args.train_data, args.test_data, args.model_save_path, args.num_epochs, args.batch_size, args.learning_rate)