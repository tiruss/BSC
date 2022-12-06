import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, input_dir):
        self.data_transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': torchvision.transforms.Compose([
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        # self.image_datasets = {x: torchvision.datasets.ImageFolder(input_dir, self.data_transforms[x]) for x in ['train', 'val']}
        # self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ['train', 'val']}
        self.dataloader = torchvision.datasets.ImageFolder(input_dir, self.data_transforms['train'])

    def __len__(self):
        return len(self.dataloader)

    def __getitem__(self, idx):
        return self.dataloader[idx]




if __name__ == "__main__":
    mydataset = MyDataset(input_dir="results")
    dataloader = DataLoader(mydataset, batch_size=4, shuffle=True, num_workers=4)

    for i, data in enumerate(dataloader, 0):
        print(data)