import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from sklearn import metrics
from numba import jit


def l2_distance(A, B):
    p1 = np.sum(A**2, axis=1)[:, np.newaxis]
    p2 = np.sum(B**2, axis=1)
    p3 = -2*np.dot(A, B.T)
    return p1 + p2 + p3


def l2_distance_vector(x):
    x = np.array([x]).transpose()
    return np.sqrt(np.abs(l2_norm(x, x)))


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


def roc_curve(target, pred):
    
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
    
    print(f'AUC = {auc}')
    
    
def loss_curve(epochs, losses):
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(epochs, losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
        
