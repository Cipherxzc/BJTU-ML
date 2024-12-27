from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from safetensors.torch import save_file, load_file

class HierarchicalClassifier(nn.Module):
    def __init__(self, clustering_model, num_classes=10, feature_dim=128):
        super(HierarchicalClassifier, self).__init__()
        self.clustering_model = clustering_model
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # 输出: 32x28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 输出: 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出: 64x7x7
            nn.Flatten(),  # 输出: 64*7*7 = 3136
            nn.Linear(3136, feature_dim),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(feature_dim + 1, 64),  # 拼接后的特征维度
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.encoder(x)
        with torch.no_grad():
            # 将输入数据转换为适合聚类模型的格式
            x_flat = x.view(x.size(0), -1).cpu().numpy()
            labels = self.clustering_model.predict(x_flat)
            labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(x.device)
        
        combined_features = torch.cat((features, labels), dim=1)
        output = self.fc(combined_features)
        return output
    
    def fit(self, train_loader, epochs=10, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        optimizer = optim.Adam(self.fc.parameters(), lr=3e-4)

        self.to(device)
        self.train()

        print(f"Training on {device}")

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
                for images, labels in tepoch:
                    optimizer.zero_grad()

                    images, labels = images.to(device), labels.to(device)
                    outputs = self(images)

                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    tepoch.set_postfix(loss=total_loss/len(tepoch), accuracy=100 * correct / total)

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    def predict(self, X):
        return self.forward(X)
    
    def predict_prob(self, X):
        outputs = self.forward(X)
        _, labels = torch.max(outputs, dim=1)
        return labels
    
    def save_model(self, path):
        save_file(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, device='cuda'):
        state_dict = load_file(path)
        self.load_state_dict(state_dict)
        self.to(device)
        print(f"Model loaded from {path}")