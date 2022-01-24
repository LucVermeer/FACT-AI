import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import os
import numpy as np
from inversefed.consts import imagenet_mean, imagenet_std
import cv2

class Imagenette(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        print(csv_file)
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = cv2.imread(img_name)
        targets = self.labels.iloc[idx, 1:]
        sample = {'image': image, 'target': targets}

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_mean_std(train_dataset):
    mean_train = torch.mean(train_dataset.dataset.data[train_dataset.indices], dim=0)
    std_train = torch.std(train_dataset.dataset.data[train_dataset.indices], dim=0)
    return mean_train, std_train


def _build_imagenette(csv_file="imagenette2/noisy_imagenette.csv", root_dir='/imagenette2', augmentations=False, normalize=True):
    """Define ImageNet with everything considered."""
    # Load data
    dataset = Imagenette(csv_file, root_dir)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, validset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    if imagenet_mean is None:
        data_mean, data_std = get_mean_std(trainset)
    else:
        data_mean, data_std = imagenet_mean, imagenet_std

    # Organize preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    if augmentations:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
        trainset.transform = transform_train
    else:
        trainset.transform = transform
    validset.transform = transform
    return trainset, validset

if __name__ == '__main__':
    trainset, valset = _build_imagenette()
    for i in range(3):
        for key in trainset[i]:
            print("entree!", key, trainset[i][key])