import os
import argparse

import cv2
import torch 
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm 

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
    images = images.permute(0, 3, 1, 2) / 255
    targets = torch.LongTensor(targets)
    
    normalize_transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.3, 0.3, 0.3])
    images = normalize_transform(images)

    data_size = images.shape[0]
    train_size = 0.8

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max', factor=0.5, patience=10)

    lr = 1e-3
    n_epochs = 20
    batch_n = len(train_data_loader)

    for g in optim.param_groups:
        g['lr'] = lr

    for epoch in tqdm(range(n_epochs)):
        model.train()
        ep_loss = 0
        for batch in train_data_loader:
            optim.zero_grad()
            
            predict = model(batch[0].to(device))
            loss = arcface_loss(predict, batch[1].to(device), n_classes=n_classes)
            
            ep_loss += loss.item()
            loss.backward()
            optim.step()
        
        train_predict, train_target = model.get_predict(train_data_loader, device)
        test_predict, test_target = model.get_predict(test_data_loader, device)
        
        train_target = train_target.cpu()
        test_target = test_target.cpu()
        
        train_predict = torch.argmax(train_predict, dim=1).cpu()
        test_predict = torch.argmax(test_predict, dim=1).cpu()
        
        train_metric = balanced_accuracy_score(train_target, train_predict)
        test_metric = balanced_accuracy_score(test_target, test_predict)

        scheduler.step(test_metric)
        
        print_loss_bool = (epoch+1)%(n_epochs//10) == 0 if n_epochs > 10 else True
        if print_loss_bool:
            print(f'Train Balanced Accuracy: {train_metric:.3f}')
            print(f'Test Balanced Accuracy: {test_metric:.3f}\n')
                
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        }, model_path
    )