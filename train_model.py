import os
import argparse

import cv2
import torch 
from tqdm import tqdm 
from torch.utils.data import DataLoader, TensorDataset

from config import global_params
from nn_module import DML, arcface_loss

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_folder', type=str, help='Путь к директории для обучения')
    parser.add_argument('model_path', type=str, help='Путь для сохранения обученной модели')

    args = parser.parse_args()

    train_data_folder = args.train_data_folder
    model_path = args.model_path
    
    down_width = global_params['down_width']
    down_height = global_params['down_height']
    embedding_size = global_params['embedding_size']
    class_targets = global_params['class_targets']
    n_classes = global_params['n_classes']
    
    images = []
    targets = []
    folder_names = os.listdir(f'{train_data_folder}')

    
    for folder in folder_names:
        for img_name in os.listdir(f'{train_data_folder}/{folder}'):
            img1 = cv2.imread(f'{train_data_folder}/{folder}/{img_name}')
            img1 = torch.FloatTensor(cv2.resize(img1, (down_width, down_height), interpolation=cv2.INTER_LINEAR))
            images.append(img1.unsqueeze(0))
            targets.append(class_targets[folder])

    images = torch.cat(images, axis=0)
    images = images.permute(0, 3, 1, 2)
    targets = torch.LongTensor(targets)

    data_size = images.shape[0]
    train_size = 0.85

    all_idx = torch.randperm(data_size)
    train_idx = all_idx[:int(data_size*train_size)]
    test_idx = all_idx[int(data_size*train_size):]

    train_data = TensorDataset(images[train_idx], targets[train_idx]) 
    test_data = TensorDataset(images[test_idx], targets[test_idx])

    train_data_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=10, shuffle=True)


    model = DML(embedding_size=embedding_size, n_classes=n_classes)
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1)
    loss_history = []

    lr = 1e-3
    n_epochs = 100
    batch_n = len(train_data_loader)

    for g in optim.param_groups:
        g['lr'] = lr

    model.train()
    for epoch in tqdm(range(n_epochs)):
        epoch_loss = 0
        for batch in train_data_loader:
            optim.zero_grad()
            
            predict = model(batch[0].to(device))
            loss = arcface_loss(predict, batch[1].to(device), n_classes=n_classes)
            
            loss.backward()
            optim.step()
            
            loss_history.append(loss.item())
            epoch_loss += loss.item()
            
        print_loss_bool = (epoch+1)%(n_epochs//10) == 0 if n_epochs > 10 else True
        if print_loss_bool:
            print(f'Loss: {epoch_loss/batch_n:.4f}')
            
            
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        }, model_path
    )