import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from simclr import SimCLR
from simclr_classifier import SimCLRClassifier
from dataset import FashionMNISTDataset

def main():
    train_data = pd.read_csv("../data/fashion-mnist_train.csv")
    test_data = pd.read_csv("../data/fashion-mnist_test.csv")

    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")



    # 将 ndarray 调整为可被 ResNet 接受的 tensor (3 * 244 * 244)
    transform = transforms.Compose([
        transforms.Resize(224),  # 调整图像大小为 224x224
        transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为 RGB 图像
        transforms.ToTensor(),  # 将图像转换为 torch.Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    train_dataset = FashionMNISTDataset(train_data.iloc[:256, :], transform=transform)
    test_dataset = FashionMNISTDataset(test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=4)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimCLR(out_dim=128)
    model.load_model('../models/simclr_model.safetensors')
    model = model.to(device)

    classifier = SimCLRClassifier(model, num_classes=10).to(device)
    classifier.fit(train_loader, epochs=1)
    # classifier.save_model('../models/simclr_classifier_model.safetensors')



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

    classifier = SimCLRClassifier(model, num_classes=10).to(device)
    classifier.load_model('../models/simclr_classifier_model.safetensors')
    evaluate(classifier, test_loader, device=device)

if __name__ == '__main__':
    main()