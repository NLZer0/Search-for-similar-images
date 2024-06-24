import os
import gc

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity

import numpy as np
import gradio as gr
from PIL import Image

import tools
from config import global_params

"""
Подгружаем модель, все параметры, а также вычисляем набор эмбеддингов для изображений
Все эти параметры могут использоваться в функции find_similar_images как глобальные переменные 
Такое решение было принятно в связи с тем, что я обнаружил, что torch model (DML) не может был передана в качестве аргумента
"""

def find_similar_images(input_image, num_images:float, find_into_class:bool):
    
    with torch.no_grad():
        input_image = data_transforms(input_image).unsqueeze(0)
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
    neighbor_imgs = [raw_data[i] for i in np.where(class_mask)[0]]
    neighbor_imgs = [str(neighbor_imgs[i][0]) for i in neighbor_img_idx]
                                                   
    return neighbor_imgs, class_str


def main(input_image, num_images:float, find_into_class:bool):
    similar_images = find_similar_images(input_image, num_images, find_into_class)
    return similar_images

if __name__ == '__main__':
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = 'model.pt'
    data_folder = 'test_data'
        
    down_width = global_params['down_width']
    down_height = global_params['down_height']
    embedding_size = global_params['embedding_size']
    class_targets = global_params['class_targets']
    target_classes = global_params['target_classes']
    n_classes = global_params['n_classes']

    # Подгуржаем модель, если model.pt не обнаружен, тогда модель будет обученна заново
    if model_path is None:
        print('\nОбучение модели\n')
        os.system(f'python train_model.py -train_data_folder {data_folder} -model_path model.pt')
        model_path = 'model.pt'

    print('\nЗагрузка модели\n')
    model = torch.jit.load(model_path, map_location=torch.device('cpu'))
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((global_params['down_width'], global_params['down_height'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=global_params['normalize_mean'],
            std=global_params['normalize_std']
        )
    ])

    raw_data = tools.load_data(data_folder, config=global_params, is_transform=False)
    raw_data, targets = raw_data.samples, raw_data.targets
    
    processed_data = tools.load_data(data_folder, config=global_params, is_transform=True)
    targets = torch.LongTensor(targets)
    
    # Получаем эмбеддинги для набора изображений
    train_data_loader = DataLoader(processed_data, batch_size=10, shuffle=False)
    images_hidden_states = tools.get_data_hidden_states(model, train_data_loader, device=device)

    # Определяем интерфейс Gradio
    image_input = gr.components.Image(type='pil', label="Загрузите изображение")
    num_images_input = gr.components.Slider(minimum=1, maximum=10, value=5, step=1, label="Количество похожих изображений")
    find_into_class = gr.components.Checkbox(label="Искать внутри класса", value=True)
    output_gallery = gr.components.Gallery(type="filepath", label="Похожие изображения")
    output_class = gr.components.Text(label="Класс объекта")

    gr.Interface(fn=main, 
                inputs=[image_input, num_images_input, find_into_class], 
                outputs=[output_gallery, output_class],
                title="Поиск похожих изображений",
                description="Загрузите изображение и выберите количество похожих изображений."
                ).launch()
