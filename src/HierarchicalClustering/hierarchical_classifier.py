from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from safetensors.torch import save_file, load_file
from .hierarchical_clustering import HierarchicalClustering

class HierarchicalClassifier(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512):
        super(HierarchicalClassifier, self).__init__()
        self.clustering_model = HierarchicalClustering(clusters=10, distance_metric='euclidean')
        self.fc = nn.Sequential(
            nn.Linear(feature_dim + 1, 64),  # 拼接后的特征维度
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = x.float()
        with torch.no_grad():
            # 将输入数据转换为适合聚类模型的格式
            x_flat = x.view(x.size(0), -1).cpu().numpy()
            labels = self.clustering_model.predict(x_flat)
            labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(x.device)
        
        combined_features = torch.cat((x, labels), dim=1)
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

    def predict(self, X, batch_size=256, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(device)
        self.eval()
        all_labels = []
        num_samples = X.shape[0]

        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="Predicting"):
                X_batch = X[i:i + batch_size]
                X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
                outputs = self.forward(X_tensor)
                _, labels = torch.max(outputs, dim=1)
                all_labels.append(labels.cpu().numpy())

        return np.concatenate(all_labels)

    def predict_proba(self, X, batch_size=256, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(device)
        self.eval()
        all_probabilities = []
        num_samples = X.shape[0]

        with torch.no_grad():
            for i in tqdm(range(0, num_samples, batch_size), desc="Predicting Probabilities"):
                X_batch = X[i:i + batch_size]
                X_tensor = torch.tensor(X_batch, dtype=torch.float32).to(device)
                outputs = self.forward(X_tensor)
                probabilities = F.softmax(outputs, dim=1)
                all_probabilities.append(probabilities.cpu().numpy())

        return np.concatenate(all_probabilities)
    
    def save_model(self, path):
        save_file(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, device='cuda'):
        state_dict = load_file(path)
        self.load_state_dict(state_dict)
        self.to(device)
        print(f"Model loaded from {path}")