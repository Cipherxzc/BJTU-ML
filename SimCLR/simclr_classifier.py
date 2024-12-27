from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from safetensors.torch import save_file, load_file



class SimCLRClassifier(nn.Module):
    def __init__(self, base_model, num_classes=10):
        super(SimCLRClassifier, self).__init__()
        self.encoder = base_model.encoder
        self.fc = nn.Linear(base_model.projector[0].in_features, num_classes)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        with torch.no_grad():
            h = self.encoder(x)
        return self.fc(h)
    
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