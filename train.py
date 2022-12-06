import torch
import torch.nn as nn
import torch.optim as optim

from classifier import Classifier

from dataloader import MyDataset

import numpy as np

# device = torch.cuda.current_device()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, input_dir):
    dataloader = torch.utils.data.DataLoader(MyDataset(input_dir=input_dir), batch_size=8, shuffle=True, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    model = model.to(device)
    for epoch in range(1000):
        print('Epoch {}/{}'.format(epoch, 100 - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    return model

if __name__ == "__main__":
    network = Classifier()
    # model.load_state_dict(torch.load('weights/classifier.pth'))
    # model.eval()
    train_model(network, input_dir="results")
