import os 

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def load_data(data_folder:str, config:dict, is_train:bool=True, batch_size:int=128):
    """
    Подгружаем данные из data_folder, разделенные по классам
    Преобразуем к тензору, и нормализуем
    """

    train_transforms = transforms.Compose([
        transforms.Resize((config['down_width'], config['down_height'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['normalize_mean'],
            std=config['normalize_std']
        )
    ])
    
    dataset = ImageFolder(data_folder, transform=train_transforms)
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, valid_loader




