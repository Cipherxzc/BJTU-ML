import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

class FashionMNISTDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data.iloc[idx, 1:].values.astype(np.uint8).reshape(28, 28)  # 不需要label
        
        image = Image.fromarray(image, mode='L')

        image1 = self.augmentation(image)
        image2 = self.augmentation(image)
        
        return image1, image2