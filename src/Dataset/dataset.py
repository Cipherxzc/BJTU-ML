from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class FeaturesDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.classes = self.data.iloc[:, -1].unique()
        self.num_classes = len(self.classes)
        self.feature_dim = self.data.shape[1] - 1  # 计算特征维度

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data.iloc[idx, :-1].values
        label = self.data.iloc[idx, -1]
        
        return feature, label


class FashionMNISTDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.classes = self.data.iloc[:, 0].unique()
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx, 1:].values.astype(np.uint8).reshape(28, 28)
        label = self.data.iloc[idx, 0]
        
        image = Image.fromarray(image, mode='L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class SimCLRDataset(Dataset):
    def __init__(self, data, augmentation):
        self.data = data
        self.augmentation = augmentation
        self.classes = self.data.iloc[:, 0].unique()
        self.num_classes = len(self.classes)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx, 1:].values.astype(np.uint8).reshape(28, 28)  # 不需要label
        
        image = Image.fromarray(image, mode='L')

        image1 = self.augmentation(image)
        image2 = self.augmentation(image)
        
        return image1, image2