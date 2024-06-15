import os
import sys 
import argparse

import cv2
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt

from config import global_params
from nn_module import DML

parser = argparse.ArgumentParser()
parser.add_argument('image_path', type=str, help='Путь к изображению для которого необходимо найти похожие')
parser.add_argument('--model_path', type=str, help='Путь к обученной модели', default=None)
parser.add_argument('--train_data_folder', type=str, help='Путь к директории для обучения', default='test_data')
parser.add_argument('--n_neighbors', type=int, help='Количество похожих изображений', default=5)

if __name__ == '__main__':
    args = parser.parse_args()
    
    assert args.image_path is not None, 'Неверно указан путь к изображению'
    image_path = args.image_path
    model_path = args.model_path
    train_data_folder = args.train_data_folder
    n_neighbors = args.n_neighbors
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    down_width = global_params['down_width']
    down_height = global_params['down_height']
    embedding_size = global_params['embedding_size']
    class_targets = global_params['class_targets']
    target_classes = global_params['target_classes']
    n_classes = global_params['n_classes']
    
    try:
        model = DML(embedding_size=embedding_size, n_classes=n_classes)
        
        if model_path is None:
            print('\nОбучение модели\n')
            os.system(f'python train_model.py -train_data_folder {train_data_folder} -model_path model.pt')
            model_path = 'model.pt'
        
        print('\nЗагрузка модели\n')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
    except:
        print("Ошибка при загрузке модели")
        sys.exit(1)

    images = []
    targets = []
    folder_names = os.listdir(train_data_folder)
    
    for folder in folder_names:
        for img_name in os.listdir(f'{train_data_folder}/{folder}'):
            img1 = cv2.imread(f'{train_data_folder}/{folder}/{img_name}')
            img1 = torch.FloatTensor(cv2.resize(img1, (down_width, down_height), interpolation=cv2.INTER_LINEAR))
            images.append(img1.unsqueeze(0))
            targets.append(class_targets[folder])

    images = torch.cat(images, axis=0)
    images = images.permute(0, 3, 1, 2)
    targets = torch.LongTensor(targets)
            
    data = TensorDataset(images, targets)
    train_data_loader = DataLoader(data, batch_size=10, shuffle=False)

    images_hidden_states = model.get_data_hidden_states(train_data_loader, device=device)

    img1 = cv2.imread(image_path)
    img1 = torch.FloatTensor(cv2.resize(img1, (down_width, down_height), interpolation=cv2.INTER_LINEAR))
    img1 = img1.permute(2, 1, 0)

    with torch.no_grad():
        img_hidden_state = model.get_embeddings(img1.unsqueeze(0).to(device)).cpu()
        img_class_predict = model(img1.unsqueeze(0).to(device)).cpu()[0]
        img_class_predict = torch.argmax(img_class_predict)
        class_str = target_classes[img_class_predict.item()]
    
    print(f'Класс объекта: {class_str}')

    class_mask = targets == img_class_predict
    similarity = cosine_similarity(img_hidden_state, images_hidden_states[class_mask])
    
    # Ищем похожие изображение на среднее первых верхних при первичном ранжировании 
    similar_imgs_hidden_state = images_hidden_states[torch.argsort(similarity)[-5:]]
    similarity = cosine_similarity(similar_imgs_hidden_state.mean(dim=0), images_hidden_states[class_mask])

    plt.figure()
    plt.title('Оригинальное изображение')
    plt.imshow((img1.permute(1,2,0)).numpy().astype(int))
    
    fig, ax = plt.subplots(nrows=n_neighbors, figsize=(12,16))
    for i, img in enumerate(images[torch.argsort(similarity)[-n_neighbors:]]):
        ax[i].imshow((img.permute(1,2,0)).numpy().astype(int))
    
    plt.show()
