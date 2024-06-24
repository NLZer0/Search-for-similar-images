import os 

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def load_data(data_folder:str, config:dict, is_transform=True):
    """
    Подгружает данные из data_folder, разделенные по классам
    Преобразует к тензору, и нормализует
    """

    train_transforms = transforms.Compose([
        transforms.Resize((config['down_width'], config['down_height'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['normalize_mean'],
            std=config['normalize_std']
        )
    ])
    
    dataset = ImageFolder(data_folder, transform=train_transforms if is_transform else None)
    return dataset 


def load_train_valid(data_folder:str, config:dict, batch_size:int=10):
    """
    Разделяет данные на train/valid 
    """

    dataset = load_data(data_folder, config)

    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader


def get_data_hidden_states(model, data_loader:DataLoader, device:torch.device):
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        image_hidden_states = []
        for batch in data_loader:
            image_hidden_states.append(model.get_embeddings(batch[0].to(device)).cpu())

        image_hidden_states = torch.cat(image_hidden_states)
    
    return image_hidden_states
                
    
def get_predict(model, data_loader:DataLoader, device:torch.device):
    model.eval()
    model.to(device)
    
    predict = []
    targets = []
    with torch.no_grad():
        for batch in data_loader:
            predict.append(
                model(batch[0].to(device))
            )
            targets.append(batch[1])
            
    predict = torch.cat(predict)
    predict /= predict.norm(p=2, dim=1)[:, None]
    targets = torch.cat(targets)
    
    return predict, targets


def get_pretrain_embeddings(model, batch:torch.Tensor):
    return model.pretrain_resnet(batch).squeeze(dim=(2,3))
    
def get_embeddings(model, batch:torch.Tensor):
    out = get_pretrain_embeddings(model, batch)
    return model.ffwd_model(out)

