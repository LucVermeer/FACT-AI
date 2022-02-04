"""
Date: 4 febuary 2022

FACT in AI
Reproducibility report for:
Privacy-preserving Collaborative Learning with Automatic Transformation Search

Students:
Dorian Bekaert
Dionne Gantzert
Ilja van Ipenburg
Luc Vermeer
"""


### Imports ###
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


### Variables ###
# Matplotlib
dot_marker = 'o'
bar_color = 'gray'
dot_color = '#1f77b4'
fig_size = (8, 5)
fontsize = 14


# Functions to load and preprocess the data
def load_data(filename, info=True):
    """
    Load the accuracy and privacy scores obtained by running
    benchmark/search_transform_attack.py.
    """

    path_accuracy = f'accuracy/{filename}/'
    path_search = f'search/{filename}/'

    results_accuracy = dict()
    results_search = dict()

    for policy in os.listdir(path_accuracy):
        results_accuracy[policy[:-4]] = np.load(f'{path_accuracy}{policy}')
        results_search[policy[:-4]] = np.load(f'{path_search}{policy}')
    
    print(f' Filename : {filename}')
    print(f' Policies : {len(results_accuracy.keys())}')

    return results_accuracy, results_search


def preprocess_data(accuracy, spri):
    """
    The accuracy and spri files contain multiple values per policy.
    For the plots of the data, the average value is used.
    """
    for policy in accuracy.keys():
        accuracy[policy] = np.mean(accuracy[policy])
        spri[policy] = np.mean(spri[policy])
    
    return accuracy, spri


def best_policies(score, n):
    """
    Print the policies with the best scores.
    Works for both privacy and accuracy scores.
    """
    return sorted(score, key=score.get)[:n]


def show_files(k):
    print(f'============ Available files for k = {k} ============')
    file_names = os.listdir(f'accuracy/k{k}')
    for i, name in enumerate(file_names):
        print(f' {i}. {name}')
    print('===================================================')
    return file_names


# Functions to plot the results
def plot_spri(k, spri, debug=True):
    """
    Plot the privacy score of the policies.

    Arguments:
    k - The policies contain at most k transformation functions.

    spri - The privacy scores per policy.

    debug - If True, print policy if a non-default type is detected.
    """

    fig = plt.figure(figsize=fig_size)    

    for i, (policy, score) in enumerate(spri.items()):
        if score.dtype != 'float64':
            if debug:
                print(f'{score.dtype} type detected in policy {policy}.')
        
        elif k == 1:
            plt.bar(int(policy), score, color=bar_color)

        else:
            plt.plot(i, score, dot_marker, color=dot_color)
    
    if k == 1:
        for policy in best_policies(spri, 5):        
            plt.bar(int(policy), spri[policy], color='red')    

    base = 0.05
    y_bottom = base * np.floor(min(list(spri.values())) / base)
    y_top = base * np.ceil(max(list(spri.values())) / base) + 0.01
    plt.ylim(y_bottom, y_top)
    plt.yticks(np.arange(y_bottom, y_top, step=base), fontsize=fontsize)
    plt.xticks(fontsize=fontsize)

    plt.xlabel('Transformation Index', size=fontsize)
    plt.ylabel(r'$S_{pri}$', size=fontsize)
    plt.show()


def plot_accuracy(k, accuracy, debug=True):
    """
    Plot the accuracy score of the policies.

    Arguments:
    k - The policies contain at most k transformation functions.

    accuracy - The accuracy scores per policy.

    debug - If True, print policy if a non-default type is detected.
    """

    fig = plt.figure(figsize=fig_size)

    for i, (policy, score) in enumerate(accuracy.items()):
        if score.dtype != 'float64' or score < -100:
            if debug:
                print(f'{score.dtype} type and {score}% accuracy detected in policy {policy}.')
            
        elif k == 1:
            plt.plot(int(policy), score, dot_marker, color=dot_color)

        else:
            plt.plot(i, score, dot_marker, color=dot_color)

    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.xlabel('Policies', size=fontsize)
    plt.ylabel('Accuracy score', size=fontsize)
    plt.title(f'Accuracy score per policy for k={k}')
    plt.show()


def plot_correlation(k, accuracy, spri, debug=True):
    """
    Plot the accuracy score of the policies.

    Arguments:

    k - The policies contain at most k transformation functions.

    accuracy - The accuracy scores per policy.

    spri - The privacy scores per policy.

    debug - If True, print policy if a non-default type is detected.
    """

    fig = plt.figure(figsize=fig_size)

    for policy, acc_score in accuracy.items():
        pri_score = spri[policy]
    
        if acc_score.dtype != 'float64' or pri_score.dtype != 'float64' or acc_score < -100:
            if debug:
                print(f'{acc_score.dtype} and {pri_score.dtype} type and {acc_score}% accuracy detected in policy {policy}.')

        else:
            plt.plot(pri_score, acc_score, dot_marker, color=dot_color)


    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.xlabel(r'$S_{pri}$', size=fontsize)
    plt.ylabel(r'$S_{acc}$', size=fontsize)
    plt.title(f'Accuracy score per policy for k={k}')
    plt.show()


def count_importance(spri, debug=False):
    counter = {t:[] for t in range(50)}

    for policy, score in spri.items():
        if score.dtype != 'float64':
            if debug:
                print(f'{score.dtype} type detected in policy {policy}.')
        
        else:
            for t in policy.split('-'):
                counter[int(t)].append(score)

    counter = {t : np.mean(score) for t, score in counter.items()}

    keys = []
    values = []

    for t, score in counter.items():
        keys.append(t)
        values.append(np.mean(score))

    fig = plt.figure(figsize=fig_size)
    plt.plot(keys, values, dot_marker, color=dot_color)

    for policy in best_policies(counter, 5):
        print(policy)
        plt.text(policy, counter[policy], policy,
                 verticalalignment='bottom', horizontalalignment='left')
        plt.plot(policy, counter[policy], dot_marker, color='red')

    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.xlabel(f'Transformation Index', size=fontsize)
    plt.ylabel(r'Average $S_{pri}$', size=fontsize)
    plt.show()
