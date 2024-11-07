import os
import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import v2, InterpolationMode, ToPILImage

def create_folder_dataset(dataset, name, transform = None):
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
        
        dataset = ImageFolder(root=dataset_directory)