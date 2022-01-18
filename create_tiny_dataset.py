import os, sys
import torch
import torchvision
seed=23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random
random.seed(seed)

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import inversefed
import torchvision.transforms as transforms
import argparse
from autoaugment import SubPolicy
from inversefed.data.data_processing import _build_cifar100, _get_meanstd
import torch.nn.functional as F
from benchmark.comm import create_model, build_transform, preprocess, create_config
from torch.utils.data import SubsetRandomSampler
from torchvision.datasets import CIFAR100
from torch.utils.data import random_split
import torch.utils.data as data



parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
opt = parser.parse_args()

# init env
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative'); 

def create_tiny_cifar100():
    train_dataset, val_dataset = _build_cifar100('/scratch/', augmentations=False, normalize=False)

    validation_size = 500
    batch_size = 32

    train_dataset, _ = random_split(train_dataset,
                                    lengths=[len(train_dataset) - validation_size, validation_size],
                                    generator=torch.Generator().manual_seed(42))
    _, val_dataset = random_split(val_dataset, 
                                  lengths=[len(val_dataset) - validation_size, validation_size],
                                  generator=torch.Generator().manual_seed(42))
    
    trainloader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validloader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    trainloader = trainloader[:5000]
    validloader = validloader[:1000]

    return trainloader, validloader

# def main():
#     if opt.data == 'fmnist':
#         trainset = torchvision.datasets.FashionMNIST('../data', train=True, download=True,
#                         transform=transforms.Compose([
#                             lambda x: transforms.functional.to_grayscale(x, num_output_channels=3),
#                             transforms.Resize(32), 
#                             transforms.ToTensor(),
#                             transforms.Normalize((0.1307,), (0.3081,))
#                         ]))
#         dataset_indices = list(range(len(trainset)))
#         dataset_indices = dataset_indices[2000:3000]
#         sampler = SubsetRandomSampler(dataset_indices)
#         trainloader = torch.utils.data.DataLoader(trainset, batch_size=defs.batch_size,
#                         drop_last=False, num_workers=4, pin_memory=True, sampler=sampler)
    

#     elif opt.data == 'cifar100':
#         downloaded_list = [
#             ['train', '16019d7e3df5f24257cddd939b257f8d'],
#         ]
#         root = os.path.join(os.getenv("HOME"), 'data')
#         base_folder = 'cifar-100-python'

#         # now load the picked numpy arrays
#         data = list()
#         targets = list()
#         for file_name, checksum in downloaded_list:
#             file_path = os.path.join(root, base_folder, file_name)
#             with open(file_path, 'rb') as f:
#                 if sys.version_info[0] == 2:
#                     entry = pickle.load(f)
#                 else:
#                     entry = pickle.load(f, encoding='latin1')
#                 data.append(entry['data'])
#                 if 'labels' in entry:
#                     targets.extend(entry['labels'])
#                 else:
#                     targets.extend(entry['fine_labels'])


# if __name__ == '__main__':
#     main()