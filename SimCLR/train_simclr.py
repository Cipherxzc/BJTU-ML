import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from simclr import SimCLR
from dataset import SimCLRDataset

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

    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(size=28, scale=(0.7, 1.0)),  # 随机裁剪并调整大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(degrees=30),  # 随机旋转，最大旋转角度为30度
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 轻微的亮度和对比度变化
        transform
    ])

    train_dataset = SimCLRDataset(train_data.iloc[:256, :], augmentation)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimCLR(out_dim=128).to(device)
    model.fit(train_loader, epochs=1, device=device)
    # model.save_model('../models/simclr_model.safetensors')



if __name__ == '__main__':
    main()