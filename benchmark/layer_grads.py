import os, sys
sys.path.insert(0, './')
import inversefed
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
from inversefed.data.loss import LabelSmoothing
from inversefed.utils import Cutout
import torch.nn.functional as F
import policy
from benchmark.comm import create_model, build_transform, preprocess, create_config


parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--aug_list', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--optim', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
parser.add_argument('--rlabel', default=False, type=bool, help='rlabel')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')
parser.add_argument('--resume', default=0, type=int, help='rlabel')
parser.add_argument('--file_name', default='figure_6_no_transform', type=str, help='File save name')
opt = parser.parse_args()
print(opt)
num_images = 1


# init env
setup = inversefed.utils.system_startup()
defs = inversefed.training_strategy('conservative'); defs.epochs = opt.epochs


# init training
arch = opt.arch
trained_model = True
mode = opt.mode
assert mode in ['normal', 'aug', 'crop']

config = create_config(opt)

def cal_dis(a, b, metric='L2'):
    a, b = a.flatten(), b.flatten()
    if metric == 'L2':
        return torch.mean((a - b) * (a - b)).item()
    elif metric == 'L1':
        return torch.mean(torch.abs(a-b)).item()
    elif metric == 'cos':
        return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    else:
        raise NotImplementedError

def create_save_dir():
    return 'benchmark/images/data_{}_arch_{}_epoch_{}_optim_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.epochs, opt.optim, opt.mode, \
        opt.aug_list, opt.rlabel)


def reconstruct(idx, model, loss_fn, trainloader, validloader):

    if opt.data == 'cifar100':
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    elif opt.data == 'FashionMinist':
        dm = torch.Tensor([0.1307]).view(1, 1, 1).cuda()
        ds = torch.Tensor([0.3081]).view(1, 1, 1).cuda()
    else:
        raise NotImplementedError

    # prepare data
    ground_truth, labels = [], []
    while len(labels) < num_images:
        img, label = validloader.dataset[idx]
        idx += 1
        if label not in labels:
            labels.append(torch.as_tensor((label,), device=setup['device']))
            ground_truth.append(img.to(**setup))

    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    param_list = [param for param in model.parameters() if param.requires_grad]

    layer_names = ['layers.0.0.conv1.weight', 'layers.0.1.conv2.weight', 'layers.1.0.conv2.weight', 'layers.2.2.conv2.weight']

    layer_params = []
    for name, param in model.named_parameters():
        if name in layer_names:
            layer_params.append(param)

    layer_grads = torch.autograd.grad(target_loss, layer_params, create_graph=True)

    input_gradient = torch.autograd.grad(target_loss, param_list)


    # attack
    print('ground truth label is ', labels)
    rec_machine = inversefed.GradientReconstructor_2(model, (dm, ds), config, num_images=num_images)
    shape = (3, 32, 32)

    if opt.rlabel:
        grad_sims = rec_machine.reconstruct(input_gradient, None, layer_grads, img_shape=shape) # reconstruction label
    else:
        grad_sims = rec_machine.reconstruct(input_gradient, labels, layer_grads, img_shape=shape) # specify label

    grad_sims = np.array(grad_sims)


    for name in layer_names:
        np.save(f'checkpoints/{opt.file_name}.npy', grad_sims)



def create_checkpoint_dir():
    return 'checkpoints/data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.mode, opt.aug_list, opt.rlabel)


def main():
    global trained_model
    print(opt)
    loss_fn, trainloader, validloader = preprocess(opt, defs, valid=True)
    model = create_model(opt)
    model.to(**setup)
    if opt.epochs == 0:
        trained_model = False
        
    if trained_model:
        checkpoint_dir = create_checkpoint_dir()
        model.load_state_dict(torch.load('checkpoints/tiny_data_cifar100_arch_ResNet20-4_mode_normal_auglist_[]_rlabel_False/ResNet20-4_50.pth'))

    if opt.rlabel:
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = False

    model.eval()
    sample_list = [i for i in range(100)]
    metric_list = list()
    mse_loss = 0
    idx = 0
    print('attach {}th in {}'.format(idx, opt.aug_list))
    metric = reconstruct(idx, model, loss_fn, trainloader, validloader)



if __name__ == '__main__':
    main()
