from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights
from safetensors.torch import save_file, load_file


class NTXentLoss(nn.Module):
    def __init__(self, temperature):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        N = z_i.shape[0]

        z = torch.cat((z_i, z_j), dim=0)
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        mask = torch.eye(N * 2, dtype=torch.bool).to(sim.device)  # 避免计算和自身的相似度
        sim = sim.masked_fill(mask, -1e9)
        
        labels = torch.arange(N)
        labels = torch.cat((labels + N, labels), dim=0).to(sim.device)

        loss = self.criterion(sim, labels)
        return loss / (N * 2)


class SimCLR(nn.Module):
    def __init__(self, out_dim):
        super(SimCLR, self).__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.projector = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
        self.encoder.fc = nn.Identity()  # num_classes 实际没有被使用

        self.criterion = NTXentLoss(temperature=0.5)

    def forward(self, x):
        features = self.encoder(x)
        return features, self.projector(features)

    def fit(self, train_loader, epochs=10, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        optimizer = optim.Adam(self.parameters(), lr=3e-4)

        self.to(device)
        self.train()

        print(f"Training on {device}")

        for epoch in range(epochs):
            total_loss = 0
            with tqdm(train_loader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{epochs}")
                for images1, images2 in tepoch:
                    optimizer.zero_grad()

                    images1, images2 = images1.to(device), images2.to(device)
                    _, z1 = self.forward(images1)
                    _, z2 = self.forward(images2)

                    loss = self.criterion(z1, z2)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    tepoch.set_postfix(loss=total_loss/len(tepoch))

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    def save_model(self, path):
        save_file(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path, device='cuda'):
        state_dict = load_file(path)
        self.load_state_dict(state_dict)
        self.to(device)
        print(f"Model loaded from {path}")