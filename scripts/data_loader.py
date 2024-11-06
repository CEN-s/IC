import os
import torch
import cv2
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2, InterpolationMode

def create_folder_dataset(dataset, num_class, num_images, transform = None):
    if transform:
        t = transform
    else:
        t = v2.Compose([
        v2.Grayscale(num_output_channels=1),
        v2.Resize((128, 128), interpolation=InterpolationMode.BILINEAR), 
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)])

    for i in range(0, num_class):
        os.Path('../datasets/'+dataset.name+'/'+dataset.name+'1'+'/'+"{:03d}".format(i)).mkdir()
        for j in range(0, num_images):
            image, _ = dataset.get_image(i+1, j+1)
            cv2.imwrite('../datasets/'+dataset.name+'/'+dataset.name+'1'+'/'+"{:03d}".format(i)+"/{:03d}".format(j)+".png", image)
    dataset = ImageFolder(root='../datasets/'+dataset.name+'/'+dataset.name+'1', transform = t)
    return dataset
        