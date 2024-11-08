from typing import overload, Union, List
import os
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import ToPILImage, InterpolationMode
from PIL import Image

@overload
def create_folder_dataset(dataset: List[tuple], name: str, transform=None) -> ImageFolder:
    ...

@overload
def create_folder_dataset(directory: str, name: str, n: int, transform=None) -> ImageFolder:
    ...

def create_folder_dataset(dataset_or_directory: Union[List[tuple], str], name: str, n: int = None, transform=None):
    # Se for uma lista de tuplas, trata como `dataset`
    if isinstance(dataset_or_directory, list):
        dataset = dataset_or_directory
        if transform:
            t = transform
        else:
            t = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((128, 128), interpolation=InterpolationMode.BILINEAR), 
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.type(torch.float32))
            ])

        dict_dataset = {}
        targets = []

        for image, target in dataset:
            if target not in targets:
                targets.append(target)
                dict_dataset[target] = [image]
            else:
                dict_dataset[target].append(image)

        dataset_directory = f"datasets/{name}/{name}1"
        if not os.path.exists(dataset_directory):
            os.makedirs(dataset_directory)

        for target, images in dict_dataset.items():
            target_directory = f"{dataset_directory}/{target}"
            if not os.path.exists(target_directory):
                os.makedirs(target_directory)
            
            for index, image in enumerate(images):
                if not isinstance(image, Image.Image):
                    image = ToPILImage()(image)
                image.save(f'{target_directory}/{index}.png')
        
        dataset = ImageFolder(root=dataset_directory, transform=t)
        return dataset
    
    # Se for um diretório, trata como `directory`
    elif isinstance(dataset_or_directory, str) and n is not None:
        directory = dataset_or_directory
        dataset = []
        files = os.listdir(directory)
        for idx, file in enumerate(files):
            image = Image.open(f'{directory}/{file}')
            label = (idx % n) + 1 if (idx % n) + 1 != 0 else n
            dataset.append((image, label))
        
        # Chama a função novamente usando o dataset carregado
        return create_folder_dataset(dataset, name, transform=transform)
    else:
        raise ValueError("Invalid parameters for create_folder_dataset.")
