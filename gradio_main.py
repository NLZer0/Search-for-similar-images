import os
import gc

import cv2
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import cosine_similarity

import numpy as np
import gradio as gr

from config import global_params
from nn_module import DML

"""
Подгружаем модель, все параметры, а также вычисляем набор эмбеддингов для изображений
Все эти параметры могут использоваться в функции find_similar_images как глобальные переменные 
Такое решение было принятно в связи с тем, что я обнаружил, что torch model (DML) не может был передана в качестве аргумента
"""

def find_similar_images(input_image, num_images:float, find_into_class:bool):
    input_image = torch.FloatTensor(cv2.resize(input_image, (down_width, down_height), interpolation=cv2.INTER_LINEAR))
    input_image = input_image.unsqueeze(0).permute(0, 3, 1, 2) / 255 
    input_image = normalize_transform(input_image)

    with torch.no_grad():
        img_hidden_state = model.get_embeddings(input_image.to(device)).cpu()
        img_class_predict = model(input_image.to(device)).cpu()[0]
        img_class_predict = torch.argmax(img_class_predict)
        class_str = target_classes[img_class_predict.item()]
    
    print(f'Класс объекта: {class_str}')
    
    if find_into_class:
        class_mask = targets == img_class_predict
    else:    
        class_mask = [True]*len(targets)
    
    similarity = cosine_similarity(img_hidden_state, images_hidden_states[class_mask])
    
    # Ищем похожие изображение на среднее первых верхних при первичном ранжировании 
    similar_imgs_hidden_state = images_hidden_states[class_mask][torch.argsort(similarity, descending=True)[:num_images]]
    similarity = cosine_similarity(similar_imgs_hidden_state.mean(dim=0), images_hidden_states[class_mask])
    
    neighbor_img_idx = torch.argsort(similarity, descending=True)[:num_images]
    neighbor_imgs = base_imgs[class_mask][neighbor_img_idx]
                                                   
    return neighbor_imgs, class_str


def main(input_image, num_images:float, find_into_class:bool):
    similar_images = find_similar_images(input_image, num_images, find_into_class)
    return similar_images

if __name__ == '__main__':
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = 'model.pt'
    train_data_folder = 'test_data'
        
    down_width = global_params['down_width']
    down_height = global_params['down_height']
    embedding_size = global_params['embedding_size']
    class_targets = global_params['class_targets']
    target_classes = global_params['target_classes']
    n_classes = global_params['n_classes']

    # Подгуржаем модель, если model.pt не обнаружен, тогда модель будет обученна заново
    model = DML(embedding_size=embedding_size, n_classes=n_classes)
    if model_path is None:
        print('\nОбучение модели\n')
        os.system(f'python train_model.py -train_data_folder {train_data_folder} -model_path model.pt')
        model_path = 'model.pt'

    print('\nЗагрузка модели\n')
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Подгружаем дамп изображений
    images = []
    targets = []
    folder_names = os.listdir(train_data_folder)

    for folder in folder_names:
        for img_name in os.listdir(f'{train_data_folder}/{folder}'):
            img1 = cv2.imread(f'{train_data_folder}/{folder}/{img_name}')
            img1 = torch.FloatTensor(cv2.resize(img1, (down_width, down_height), interpolation=cv2.INTER_LINEAR))
            images.append(img1.unsqueeze(0))
            targets.append(class_targets[folder])

    images = torch.cat(images, axis=0).permute(0, 3, 1, 2) / 255
    targets = torch.LongTensor(targets)

    base_imgs = np.array([
        torchvision.transforms.functional.to_pil_image(it, mode=None) for it in images
    ]) # сохраняем исходные изображения без нормализации для итогового вывода

    normalize_transform = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.3, 0.3, 0.3])
    images = normalize_transform(images)
    
    # Получаем эмбеддинги для набора изображений
    data = TensorDataset(images, targets)
    train_data_loader = DataLoader(data, batch_size=10, shuffle=False)
    images_hidden_states = model.get_data_hidden_states(train_data_loader, device=device)
    
    # Набор нормализованных изображений больше не используется, поэтому освобождаем память
    del images
    gc.collect()

    
    # Определяем интерфейс Gradio
    image_input = gr.components.Image(label="Загрузите изображение")
    num_images_input = gr.components.Slider(minimum=1, maximum=10, value=5, step=1, label="Количество похожих изображений")
    find_into_class = gr.components.Checkbox(label="Искать внутри класса", value=True)
    output_gallery = gr.components.Gallery(type="pil", label="Похожие изображения")
    output_class = gr.components.Text(label="Класс объекта")

    gr.Interface(fn=main, 
                inputs=[image_input, num_images_input, find_into_class], 
                outputs=[output_gallery, output_class],
                title="Поиск похожих изображений",
                description="Загрузите изображение и выберите количество похожих изображений."
                ).launch()
