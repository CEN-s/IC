import os
import torch
import cv2
from torchvision.datasets import ImageFolder, Subset, ConcatDataset, DataLoader
from rnn import RNN
from splitter import WindowSplitter

def fit(model, data, device, indices, num_epochs=100, aug_factor=1):
    print(f"Augmentation factor: {1 if aug_factor == 0 else aug_factor}")
    splitter = WindowSplitter()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    rnn = RNN(Q=aug_factor, P=3*3)
    
    curr_aug = len([f for f in os.listdir(os.dirname(data.root)) if os.isdir(os.irname(data.root))])
    
    if aug_factor > curr_aug:
        for n in range(curr_aug+1, aug_factor+1):
            if not os.Path(f'../datasets/Outex/Outex{n}/').is_dir():
                os.Path(f'../datasets/Outex/Outex{n}/').mkdir()
            for i in range(0,68):
                if not os.Path(f'../datasets/Outex/Outex{n}/'+"{:03d}".format(i)).is_dir():
                    os.Path(f'../datasets/Outex/Outex{n}/'+"{:03d}".format(i)).mkdir()

        j = 0 
        for image, label in data:
            input = splitter.split(image.reshape(128,128), window_size=3, padding=True)
            images = rnn._generate_images(input)
            images = images.reshape(aug_factor, 1, 128, 128)
            images =  (images*255).astype(np.uint8)
            for n in range(curr_aug+1, aug_factor+1):
                cv2.imwrite(f'../datasets/Outex/Outex{n}/'+"{:03d}".format(label)+"/{:03d}".format(j)+".png", images[n-1].reshape(128,128))
                j = j + 1      

    train = Subset(data, indices[0])
    val = Subset(data, indices[1])
    test = Subset(data, indices[2])

    dataset = []
    for i in range(1, aug_factor+1):
        if i == 1:
            dataset.append(train)
            continue
        dataset.append(Subset(ImageFolder(root=f"../datasets/Outex/Outex{i}", transform = t), indices[0]))
    train = ConcatDataset(dataset)

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