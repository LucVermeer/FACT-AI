import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import os
import numpy as np
from inversefed.consts import imagenet_mean, imagenet_std
import matplotlib.pyplot as plt
from PIL import Image

class Imagenette(Dataset):
    def __init__(self, csv_file, root_dir, transform = None):
        # print(csv_file)
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])

    def __len__(self):
        return len(self.labels)

    def target_one_hot(self, string):
        list = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
        if string in list:
            return list.index(string)
        return None

    def __getitem__(self, idx):
        # print('getting item', idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print('---------------------------------\ngetting item')
        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name)
        image = self.transform(image)
        if image.shape[0] == 1:
            gray = image.squeeze()
            # gray = torch.div(gray, 3, rounding_mode="floor")
            image = torch.stack([gray, gray, gray], dim=0)
        targets = self.labels.iloc[idx, 1]
        targets = self.target_one_hot(targets)
        sample = (image, targets)
        return sample


def get_mean_std(train_dataset):
    mean_train = torch.mean(train_dataset.dataset.data[train_dataset.indices], dim=0)
    std_train = torch.std(train_dataset.dataset.data[train_dataset.indices], dim=0)
    return mean_train, std_train


def _build_imagenette(csv_file="../../scratch/noisy_imagenette.csv", root_dir='../../scratch/', augmentations=False, normalize=True):
    """Define ImageNet with everything considered."""
    # Load data
    dataset = Imagenette(csv_file, root_dir)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, validset = torch.utils.data.random_split(dataset, [train_size, test_size])

    

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
    validset.transform = transform
    if augmentations:
        # print('augmentations')
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x : x)])
    trainset.transform = transform
    # print(trainset.transform)
    return trainset, validset

if __name__ == '__main__':
    trainset, valset = _build_imagenette()
    print(trainset.transform)
    for i in range(3):
        for key in trainset[i]:
            print("entry", key, trainset[i][key])