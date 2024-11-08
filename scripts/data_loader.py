import os
import torch
from typing import overload
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import v2, InterpolationMode, ToPILImage
from torch.utils.data import Dataset

class ListDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.dataset = data_list
        self.transform = transform


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.data_list[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

@overload
def create_folder_dataset(dataset, name, transform=None):
    if transform:
        t = transform
    else:
        t = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((128, 128), interpolation=InterpolationMode.BILINEAR), 
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)])

        dict_dataset = {}
        targets = []

        for image, target in dataset:
            if target not in targets:
                targets.append(target)
                dict_dataset[target] = [image]
            else:
                dict_dataset[target].append(image)

        dataset_directory = f"datasets/{name}/{name}1"
        if os.path.exists(dataset_directory):
            os.makedirs(dataset_directory)

        for target, images in dataset_directory.items():
            target_directory = f"{dataset_directory}/{target}"
            if os.path.exist(target_directory):
                os.makedirs(target_directory)
            
            for image in images:
                image = ToPILImage(image)
                image.save(f'{target_directory}/{images.index(image)}')
        
        dataset = ImageFolder(root=dataset_directory, transform=t)

@overload
def create_folder_dataset(directory, name, labels, n, transform=None):
    dataset = []
    files = os.listdir(directory)
    for file in files:
        image = Image.open(f'{directory}/{file}')
        label = (files.index(file) % n) + 1 if (files.index(file) % n) + 1 != 0 else n
        dataset.append((image, label))
    dataset = ListDataset(dataset, transform)
    dataset = create_folder_dataset(dataset, name, transform)
    return dataset

        