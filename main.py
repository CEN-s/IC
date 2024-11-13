import os

import scripts.reproducibility as rep
from scripts.data_loader import create_folder_dataset
from scripts.fitter import fit

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import torch
import timm
import pandas as pd
import matplotlib.pyplot as plt 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

rep.set_config()

t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((128, 128), interpolation=InterpolationMode.BILINEAR), 
                transforms.Lambda(lambda x: x.type(torch.float32))
            ])

if not os.path.exists("datasets/1200tex"):
    data = create_folder_dataset("datasets originais/1200tex", "1200tex", 60)

data = ImageFolder(root="datasets/1200tex/1200tex1", transform=t)

indices = rep.split_indices(data)

accuracies = []
for i in range(1, 11):
    resnet = timm.create_model('resnet18', pretrained=False, in_chans=1, num_classes=20)
    resnet.to(device)
    accuracies.append(fit(resnet, data, device, indices, aug_factor=i,  num_epochs=100))

accuracies = pd.DataFrame(accuracies)
accuracies.to_csv('resultados/accuracy_1200tex.csv')
accuracies.columns = ['Accuracy']
accuracies.index = range(1, len(accuracies) + 1)
accuracies.plot()
plt.xlabel('Augmentation Factor')
plt.ylabel('Accuracy')
plt.title('Model Accuracy vs Augmentation Factor')
plt.savefig('resultados/accuracy_plot_1200tex.png')