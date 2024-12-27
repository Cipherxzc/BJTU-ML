import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from src.SimCLR.simclr import SimCLR
from src.Dataset.dataset import FashionMNISTDataset, SimCLRDataset
from torchvision.models import resnet18
from safetensors.torch import load_file

# 加载预训练的 SimCLR 和 ResNet 模型
simclr_model_path = 'models/simclr_model.safetensors'
resnet_model_path = 'models/resnet_model.safetensors'

simclr = SimCLR()
simclr.load_model(simclr_model_path)
simclr.eval()

resnet = resnet18()
resnet.fc = torch.nn.Identity()  # 去掉最后一层
resnet.load_state_dict(load_file(resnet_model_path))
resnet.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
simclr.to(device)
resnet.to(device)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data_path = "data/fashion-mnist_train.csv"
test_data_path = "data/fashion-mnist_test.csv",

train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

train_dataset = FashionMNISTDataset(train_data, transform=transform)
test_dataset = FashionMNISTDataset(test_data, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

# 生成特征集1 (features1) 使用 ResNet
features1_train = []
features1_test = []

with torch.no_grad():
    for images, _ in train_loader:
        images = images.to(device)
        features = resnet(images)
        features1_train.append(features.cpu())

    for images, _ in test_loader:
        images = images.to(device)
        features = resnet(images)
        features1_test.append(features.cpu())

features1_train = torch.cat(features1_train, dim=0).numpy()
features1_test = torch.cat(features1_test, dim=0).numpy()

# 保存特征集1
pd.DataFrame(features1_train).to_csv('features1_train.csv', index=False)
pd.DataFrame(features1_test).to_csv('features1_test.csv', index=False)

print("Feature extraction completed and saved to CSV files.")
