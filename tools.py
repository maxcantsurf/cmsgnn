import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import tkinter as tk

from tkinter import filedialog
from sklearn import metrics
from numba import jit


rc = {'font.size': 18, 
      'mathtext.fontset': 'cm',
      'legend.fontsize': 18}
plt.rcParams.update(rc)


def l2_norm(A, B):
    p1 = np.sum(A**2, axis=1)[:, np.newaxis]
    p2 = np.sum(B**2, axis=1)
    p3 = -2*np.dot(A, B.T)
    return p1 + p2 + p3


@jit(nopython = True)
def k_nn_graph(A, k):
    
    # Takes in an adjacency matrix, and removes edges, keeping only the 
    # edges which connect the closest k-neiighbours to each vertex
    # Returns the new adjacency matrix and an array listing the nodes that
    # have been left connected to each node
    
    # Probably a much faster way to do this
    
    n = len(A)
    A += 255*np.identity(n)
    A_new = np.zeros((n, n))
    indices_full = np.zeros((n, k), dtype=np.uint8)
    
    for i in range(n):
        indices = np.argsort(A[i])[:k]
        j = 0
        
        for index in indices:
            A_new[i][index] = A[i][index]
            A_new[index][i] = A[index][i]
            indices_full[i][j] = index
            j += 1
    
    for i in range(n):
        A_new[i][i] = 0
    
    return A_new, indices_full


def roc_curve(target, pred, name=None, model=None, cluster=None):
    
    fpr, tpr, thresholds = metrics.roc_curve(target, pred)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(fpr, tpr)
    ax.grid()
    ax.set_xscale('log')
    ax.set_xlabel('FPR / Mistag Rate')
    ax.set_ylabel('TPR / Efficiency')
    
    auc = metrics.auc(fpr, tpr)
    ax.set_title(f'AUC = {auc}')
    
    if name is not None:
        
        data = {'model':model,
                'name':name,
                'cluster':cluster,
                'fpr':fpr,
                'tpr':tpr,
                'auc':auc
                }
        
        with open(os.getcwd() + '/rocs/'+ name + '.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
    print(f'AUC = {auc}')


def loss_curve(epochs, losses):
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')


def plot_rocs():
    root = tk.Tk()
    root.withdraw()
    inititaldir = os.getcwd() + '/rocs/'
    files = filedialog.askopenfilenames(parent=root, initialdir=inititaldir, 
                                        title='Please select files')
    runs = []
    
    for file in files:
        with open(file, 'rb') as f:
            run = pickle.load(f)
            runs.append(run)
            
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid()
    ax.set_xscale('log')
    ax.set_xlabel('FPR / Mistag Rate')
    ax.set_ylabel('TPR / Efficiency')
    
    for run in runs:
        tpr = run['tpr']
        fpr = run['fpr']
        auc = run['auc']
        name = run['name']
    
        label = f'{name}, AUC = {round(auc, 3)}'
        ax.plot(fpr, tpr, label=label)
    
    ax.legend()
