import os
from pathlib import Path

import torch
import PIL
import numpy as np
import pandas as pd

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, ConcatDataset

from scripts.rnn import RNN
from scripts.splitter import WindowSplitter
from scripts.data_loader import create_folder_dataset   

def fit(model, data, device, indices, num_epochs=100, aug_factor=1):
    root = Path(data.root)
    parent = root.parent
    name = parent.name

    accuracies = []
    print(f"Augmentation factor: {1 if aug_factor == 0 else aug_factor}")
    splitter = WindowSplitter()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    rnn = RNN(Q=aug_factor, P=3*3)
    
    
    curr_aug = len([f for f in root.parent.iterdir()])
    if aug_factor > curr_aug:
        for n in range(curr_aug+1, aug_factor+1):
            for i in range(0,20):
                Path(parent/f'{name}{n}'/f'{i}').mkdir(parents=True, exist_ok=True)

        j = 0
        for image, label in data:
            input = splitter.split(image.reshape(128,128), window_size=3, padding=True)
            images = rnn._generate_images(input)
            images = images.reshape(aug_factor, 1, 128, 128)
            images =  (images*255)
            for i in range(curr_aug+1, aug_factor+1):
                pil_image = PIL.Image.fromarray(images[i-1].reshape(128, 128).astype(np.uint8))
                pil_image.save(parent/f'{name}{i}'/f'{label}/{j}.png')
                j = j + 1

    train = Subset(data, indices[0])
    val = Subset(data, indices[1])
    test = Subset(data, indices[2])

    dataset = []
    for i in range(1, aug_factor+1):
        if i == 1:
            dataset.append(train)
            continue
        dataset.append(Subset(ImageFolder(root=parent/f'{name}{i}', transform = data.transform), indices[0]))
    train = ConcatDataset(dataset)

    print("Train size: ", len(train))

    train_loader = DataLoader(train, batch_size=int(len(train) * 0.05), shuffle=True)
    val_loader = DataLoader(val, batch_size=int(len(val) * 0.05), shuffle=False)
    test_loader = DataLoader(test, batch_size=int(len(test) * 0.05), shuffle=False)

    
    model.train()
    max_acc = 0
    for epoch in range(num_epochs):
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.long())
            loss.backward()
            optimizer.step()

        correct_count = 0
        total_count = 0
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            correct_val= 0
            total_val = 0
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X).squeeze()
                _, predicted = torch.max(outputs.data, 1)
                total_val += batch_y.size(0)
                correct_val += (predicted == batch_y).sum().item()
            curr_acc = correct_val / total_val
            if curr_acc > max_acc:
                print(f"Epoch {epoch+1}, New max acc.: {max_acc * 100}")
                max_acc = curr_acc
                state_dict = model.state_dict()

    model.load_state_dict(state_dict)
    model.eval()
    correct_test = 0
    total_test = 0
    for batch_X, batch_y in test_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X).squeeze()
        _, predicted = torch.max(outputs.data, 1)
        total_test += batch_y.size(0)
        correct_test += (predicted == batch_y).sum().item()
    acc = correct_test / total_test
    print(f"Test acc.: {acc * 100}")
    accuracies.append(acc)
    return accuracies
 