# python -u searchalg/batch_generate.py  --arch=ResNet20-4 --data=cifar100
from copy import deepcopy
import random
import argparse
import numpy as np

# For reproducibility
random.seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
opt = parser.parse_args()

scheme_list = list()

# Variables
num_per_gpu = 4     # Number of DNNs per GPU
num_epochs = 50      # Number of epochs
k = 3
Cmax = 1500



def write():
    for i in range(len(scheme_list) // num_per_gpu):
        for id, idx in enumerate(range(i*num_per_gpu, i*num_per_gpu + num_per_gpu)):
            sch_list = [str(sch) for sch in scheme_list[idx]]
            suf = '-'.join(sch_list)

            cmd = f'python benchmark/search_transform_attack.py' + \
                  f' --aug_list={suf} --mode=aug --arch={opt.arch} --data={opt.data} --epochs={num_epochs}'

            if id != i*num_per_gpu + num_per_gpu - 1:
                cmd += '&'
            print(cmd)
        print('wait')


def backtracing(num):
    """
    TODO: Wat betekent 'num'?
    """
    for _ in range(Cmax):
        scheme = list(np.random.randint(-1, 51, k))
        new_policy = deepcopy(scheme)
        new_policy = [x for x in scheme if x >= 0]
        scheme_list.append(new_policy)
    write()


if __name__ == '__main__':
    backtracing(5)
