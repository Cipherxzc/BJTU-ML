import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.SimCLR.simclr import SimCLR
from src.Dataset.dataset import FashionMNISTDataset, SimCLRDataset
from src.Dataset.transform import Transform, Augmentation
from torchvision.models import resnet18
from safetensors.torch import load_file
from tqdm import tqdm

def main():
    # 加载预训练的 SimCLR 和 ResNet 模型
    simclr_model_path = 'models/simclr_model.safetensors'
    resnet_model_path = 'models/resnet_model.safetensors'

    simclr = SimCLR(out_dim=128)  # outdim 只在训练时有用
    simclr.load_model(simclr_model_path)
    simclr.eval()

    resnet = resnet18()
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 10)
    resnet.load_state_dict(load_file(resnet_model_path))
    resnet.fc = torch.nn.Identity()  # 去掉最后一层
    resnet.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    simclr.to(device)
    resnet.to(device)

    transform = Transform()
    augmentation = Augmentation()

    train_data_path = "data/fashion-mnist_train.csv"
    test_data_path = "data/fashion-mnist_test.csv"

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    train_dataset = FashionMNISTDataset(train_data, transform=transform)
    test_dataset = FashionMNISTDataset(test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    # 生成特征集1 (features1) 使用 ResNet
    features1_train = []
    labels_train = []
    features1_test = []
    labels_test = []

    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Processing Training Data"):
            images = images.to(device)
            features = resnet(images)
            features1_train.append(features.cpu())
            labels_train.append(labels)

        for images, labels in tqdm(test_loader, desc="Processing Test Data"):
            images = images.to(device)
            features = resnet(images)
            features1_test.append(features.cpu())
            labels_test.append(labels)

    features1_train = torch.cat(features1_train, dim=0).numpy()
    labels_train = torch.cat(labels_train, dim=0).numpy()
    features1_test = torch.cat(features1_test, dim=0).numpy()
    labels_test = torch.cat(labels_test, dim=0).numpy()

    # 保存特征集1
    train_df = pd.DataFrame(features1_train)
    train_df['label'] = labels_train
    train_df.to_csv('data/features1_train.csv', index=False)

    test_df = pd.DataFrame(features1_test)
    test_df['label'] = labels_test
    test_df.to_csv('data/features1_test.csv', index=False)

    print(train_df)
    print(test_df)

    print("ResNet Feature extraction completed and saved to CSV files.")


    train_dataset = SimCLRDataset(train_data, augmentation=augmentation)
    test_dataset = SimCLRDataset(test_data, augmentation=augmentation)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    # 生成特征集2 (features2) 使用 SimCLR
    features2_train = []
    labels_train = []
    features2_test = []
    labels_test = []

    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="Processing Training Data"):
            images = images.to(device)
            features, _ = simclr(images)
            features2_train.append(features.cpu())
            labels_train.append(labels)

        for images, labels in tqdm(test_loader, desc="Processing Test Data"):
            images = images.to(device)
            features, _ = simclr(images)
            features2_test.append(features.cpu())
            labels_test.append(labels)

    features2_train = torch.cat(features2_train, dim=0).numpy()
    labels_train = torch.cat(labels_train, dim=0).numpy()
    features2_test = torch.cat(features2_test, dim=0).numpy()
    labels_test = torch.cat(labels_test, dim=0).numpy()

    # 保存特征集2
    train_df = pd.DataFrame(features2_train)
    train_df['label'] = labels_train
    train_df.to_csv('data/features2_train.csv', index=False)

    test_df = pd.DataFrame(features2_test)
    test_df['label'] = labels_test
    test_df.to_csv('data/features2_test.csv', index=False)

    print(train_df)
    print(test_df)

    print("SimCLR Feature extraction completed and saved to CSV files.")


if __name__ == "__main__":
    main()