import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from hierarchical_clustering import HierarchicalClustering
from hierarchical_classifier import HierarchicalClassifier
from dataset import FashionMNISTDataset

def evaluation_fn(y, labels):
    return ((1 - adjusted_rand_score(y, labels)) / 2)**2 + (1 - normalized_mutual_info_score(y, labels))**2

def main():
    train_data = pd.read_csv("../data/fashion-mnist_train.csv")
    test_data = pd.read_csv("../data/fashion-mnist_test.csv")

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")


    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为 torch.Tensor
        transforms.Normalize(mean=[0.5], std=[0.5]),  # 标准化
    ])

    train_dataset = FashionMNISTDataset(train_data.iloc[:, :], transform=transform)
    test_dataset = FashionMNISTDataset(test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=4)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HierarchicalClustering.load_model("../models/hierarchical_clustering_model.pkl")

    # classifier = HierarchicalClassifier(model, num_classes=10).to(device)
    # classifier.fit(train_loader, epochs=10, device=device)
    # classifier.save_model('../models/hierarchical_classifier_model.safetensors')


    def evaluate(model, test_loader, device='cuda'):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")

    classifier = HierarchicalClassifier(model, num_classes=10).to(device)
    classifier.load_model('../models/hierarchical_classifier_model.safetensors')

    evaluate(classifier, test_loader, device=device)



if __name__ == "__main__":
    main()