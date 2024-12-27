import torchvision.transforms as transforms

def Transform():
    return transforms.Compose([
        transforms.Resize(224),  # 调整图像大小为 224x224
        transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为 RGB 图像
        transforms.ToTensor(),  # 将图像转换为 torch.Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

def Augmentation():
    return transforms.Compose([
        transforms.RandomResizedCrop(size=28, scale=(0.7, 1.0)),  # 随机裁剪并调整大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(degrees=30),  # 随机旋转，最大旋转角度为30度
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 轻微的亮度和对比度变化
        Transform()
    ])