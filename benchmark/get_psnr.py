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

opt = parser.parse_args()
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


def create_save_dir():
    return 'benchmark/images/data_{}_arch_{}_epoch_{}_optim_{}_mode_{}_auglist_{}_rlabel_{}'.format(opt.data, opt.arch, opt.epochs, opt.optim, opt.mode, \
        opt.aug_list, opt.rlabel)

def PSNR(original, compressed):
    mse = ((original - compressed) ** 2).mean()
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1.0
    # psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    psnr = 10 * torch.log10(max_pixel**2 / mse)
    return psnr

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

    input_gradient = torch.autograd.grad(target_loss, param_list)


    # attack
    print('ground truth label is ', labels)
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=num_images)
    print(rec_machine)
    if opt.data == 'cifar100':
        shape = (3, 32, 32)
    elif opt.data == 'FashionMinist':
        shape = (1, 32, 32)

    if opt.rlabel:
        output, stats = rec_machine.reconstruct(input_gradient, None, img_shape=shape) # reconstruction label
    else:
        output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=shape) # specify label

    output_denormalized = output * ds + dm
    input_denormalized = ground_truth * ds + dm

    psnr = inversefed.metrics.psnr(output_denormalized, input_denormalized)

    print("PSNR: ", psnr)

    return test_psnr




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
        model.load_state_dict(torch.load('checkpoints/tiny_data_cifar100_arch_ResNet20-4_mode_normal_auglist_[]_rlabel_False/ResNet20-4_50.pth'))
        

    if opt.rlabel:
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = False

    model.eval()
    idx = 0
    print('attach {}th in {}'.format(idx, opt.aug_list))
    test_psnr = reconstruct(idx, model, loss_fn, trainloader, validloader)
    save_dir = create_save_dir()
    np.save('{}/{}_psnr.npy'.format(save_dir, opt.aug_list), test_psnr)
    # np.save('{}/{}_psnr.npy'.format(save_dir, opt.aug_list), psnr)



if __name__ == '__main__':
    main()
