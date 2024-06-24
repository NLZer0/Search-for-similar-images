import os
import argparse

import cv2
import torch 
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm 

import tools
from config import global_params
from nn_module import DML, arcface_loss

import warnings
warnings.filterwarnings("ignore")

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
    n_classes = global_params['n_classes']
    
    train_data_loader, valid_data_loader = tools.load_train_valid(train_data_folder, config=global_params, batch_size=10)

    model = torch.jit.script(DML(embedding_size=embedding_size, n_classes=n_classes))
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=10)

    lr = 1e-3
    n_epochs = 10
    batch_n = len(train_data_loader)

    for g in optim.param_groups:
        g['lr'] = lr

    for epoch in tqdm(range(n_epochs)):
        model.train()
        ep_loss = 0
        for batch, target in train_data_loader:
            optim.zero_grad()
            
            predict = model(batch.to(device))
            loss = arcface_loss(predict, target.to(device), n_classes=n_classes)
            
            ep_loss += loss.item()
            loss.backward()
            optim.step()
        
        train_predict, train_target = tools.get_predict(model, train_data_loader, device)
        valid_predict, valid_target = tools.get_predict(model, valid_data_loader, device)
        
        train_target = train_target.cpu()
        valid_target = valid_target.cpu()
        
        train_predict = torch.argmax(train_predict, dim=1).cpu()
        valid_predict = torch.argmax(valid_predict, dim=1).cpu()
        
        train_metric = balanced_accuracy_score(train_target, train_predict)
        valid_metric = balanced_accuracy_score(valid_target, valid_predict)

        scheduler.step(valid_metric)
        
        print_loss_bool = (epoch+1)%(n_epochs//10) == 0 if n_epochs > 10 else True
        if print_loss_bool:
            print(f'Train Balanced Accuracy: {train_metric:.3f}')
            print(f'Valid Balanced Accuracy: {valid_metric:.3f}\n')
                
    torch.jit.save(
        model,
        model_path,

    )