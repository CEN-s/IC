import os
import torch
from typing import overload, Union, List
from concurrent.futures import ThreadPoolExecutor

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToPILImage, InterpolationMode

from PIL import Image

def load_image(file_path):
    image = Image.open(file_path)
    return image


def save_image(image, file_path):
    image.save(file_path)

@overload
def create_folder_dataset(dataset: List[tuple], name: str, transform=None) -> ImageFolder:
    ...

@overload
def create_folder_dataset(directory: str, name: str, n: int, transform=None) -> ImageFolder:
    ...

def create_folder_dataset(dataset_or_directory: Union[List[tuple], str], name: str, n: int = None, transform=None):
    if transform:
        t = transform
    else:
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128), interpolation=InterpolationMode.BILINEAR), 
            transforms.Lambda(lambda x: x.type(torch.float32))
        ])

    if isinstance(dataset_or_directory, list):
        dataset = dataset_or_directory
        
        dict_dataset = {}
        targets = []

        for image, target in dataset:
            if target not in targets:
                targets.append(target)
                dict_dataset[target] = [image]
            else:
                dict_dataset[target].append(image)

        dataset_directory = f"datasets/{name}/{name}1"
        os.makedirs(dataset_directory, exist_ok=True)
        for target in set(targets):
            target_directory = f"{dataset_directory}/{target}"
            os.makedirs(target_directory, exist_ok=True)

        for target, images in dict_dataset.items():            
            for index, image in enumerate(images):
                if not isinstance(image, Image.Image):
                    image = ToPILImage()(image)
                with ThreadPoolExecutor() as executor:
                    executor.submit(save_image, image, f"{dataset_directory}/{target}/{index}.png")
        
        dataset = ImageFolder(root=dataset_directory, transform=t)
        return dataset
    
    elif isinstance(dataset_or_directory, str) and n is not None:
        directory = dataset_or_directory
        files = os.listdir(directory)
        files.sort()
        with ThreadPoolExecutor() as executor:
            dataset = []
            for i, file in enumerate(files):
                file_path = f'{directory}/{file}'
                image = executor.submit(load_image, file_path).result()
                label = i // n
                dataset.append((image, label))
        
        return create_folder_dataset(dataset, name, transform=t)
    else:
        raise ValueError("Invalid parameters for create_folder_dataset.")
