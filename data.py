import numpy as np
import networkx as nx
import uproot
import torch
import tools
import yaml
import os
import torch_geometric.utils as utils
import networkx.algorithms.centrality as centrality

from tqdm import tqdm
from numba import jit
from torch_geometric.data import Data


@jit(nopython = True)
def ktmetric(kt2_i, kt2_j, dR2_ij, p = -1, R = 1.0):
    """ Taken from Icenet """

    return min([kt2_i**(2*p), kt2_j**(2*p)]) * (dR2_ij/R**2)


@jit(nopython = True)
def edgelist_to_adjmatrix(edges, edge_attr, num_nodes, num_edges):
    A = np.zeros((num_nodes, num_nodes))
    
    for n in range(num_edges):
        v0 = edges[0,n]
        v1 = edges[1,n]
        A[v0,v1] = edge_attr[n]
        
    return A


@jit(nopython = True)
def vec_to_dist_matrix(x, y):
    n = len(x)
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            D[i,j] = ((x[i]-x[j])**2 + (y[i]-y[j])**2)
    
    return D


@jit(nopython = True)
def floyd_warshall(D, maxdist = 64):
    n = len(D)
    
    for i in range(n):
        D[i,i] = 0.0
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if D[i,j] > D[i,k] + D[k,j]:
                    D[i,j] = D[i,k] + D[k,j]
    return D


@jit(nopython = True)
def farness_centrality(D, normed=False):
    n = len(D)
    f = np.zeros(n)
    
    if n == 1:
        return np.zeros(1)
    
    for i in range(n):
        f[i] = np.sum(D[i])
    
    return f
    


class DataGrabber():
    
    def __init__(self, dir_path, file_name):
        self.dir_path = dir_path
        self.file_name = file_name
        self.root_file = uproot.open(dir_path + file_name)
        self.events_full = self.root_file['ntuplizer']['tree']
        self.numentries = self.events_full.numentries
        
        with open(os.getcwd() + '/dataconfig.yml') as file:
            dataconfig = yaml.full_load(file)
        
        self.global_feature_names = dataconfig['global_feature_names']
        self.cluster_feature_names = dataconfig['cluster_feature_names']
        self.pfcluster_feature_names = dataconfig['pfcluster_feature_names']
        
        self.num_global_features = len(self.global_feature_names)
        self.num_node_features = 10
        self.num_edge_features = 12
        

    def get_dataset(self, entrystart, entrystop):
        
        self.global_features = {}
        self.cluster_features = {}
        self.pfcluster_features = {}
        
        for feature in self.global_feature_names:
            self.global_features[feature] = self.events_full[feature].array(entrystart=entrystart, entrystop=entrystop)
        
        for feature in self.cluster_feature_names:
            self.cluster_features[feature] = self.events_full[feature].array(entrystart=entrystart, entrystop=entrystop)
        
        for feature in self.pfcluster_feature_names:
            self.pfcluster_features[feature] = self.events_full[feature].array(entrystart=entrystart, entrystop=entrystop)
        
        self.is_e = self.events_full['is_e'].array(entrystart=entrystart, entrystop=entrystop)
        
        dataset = []
        
        for event in tqdm(range(len(self.is_e))):
            
            num_clu = self.global_features['image_clu_n'][event]
            num_pf = self.global_features['image_pf_n'][event]
            
            num_nodes = num_clu + num_pf
            
            # Discard events which have one cluster or less
            
            if num_clu > 0 and num_pf > 0:
                
                clu_eta = self.cluster_features['image_clu_eta'][event]
                clu_phi = self.cluster_features['image_clu_phi'][event]
                
                clu_e = self.cluster_features['image_clu_e'][event]
                clu_nhit = self.cluster_features['image_clu_nhit'][event]
                
                pf_eta = self.pfcluster_features['image_pf_eta'][event]
                pf_phi = self.pfcluster_features['image_pf_phi'][event]
                
                pf_p = self.pfcluster_features['image_pf_p'][event]
                pf_pdgid = self.pfcluster_features['image_pf_pdgid'][event]
                
                x = np.zeros((num_nodes, self.num_node_features))
                y = np.zeros(num_nodes)
                u = np.zeros((num_nodes, self.num_global_features))
                
                n = 0
                
                # TODO: Make this cleaner
                # clu node attributes
                
                for i in range(num_clu):
                    x[n][0] = clu_eta[i]
                    x[n][1] = clu_phi[i]
                    x[n][2] = clu_e[i]
                    x[n][3] = clu_nhit[i]
                    x[n][4] = 0.0
                    x[n][5] = 0.0
                    x[n][6] = 1.0
                    x[n][7] = 0.0
                    x[n][8] = 0.0
                    x[n][9] = 0.0
                    
                    u[n][0] = num_clu
                    u[n][1] = num_pf
                    
                    n += 1
                    
                # pf node attributes
                
                for i in range(num_pf):
                    x[n][0] = 0.0
                    x[n][1] = 0.0
                    x[n][2] = 0.0
                    x[n][3] = 0.0
                    x[n][4] = pf_p[i]
                    x[n][5] = pf_pdgid[i]
                    x[n][6] = 0.0
                    x[n][7] = 1.0
                    x[n][8] = pf_eta[i]
                    x[n][9] = pf_phi[i]
                    
                    u[n][0] = num_clu
                    u[n][1] = num_pf
                    
                    n += 1
                    
                # Need to connect the clu nodes together and the pf
                # nodes together such that they form two separate graphs
                
                num_edges = num_clu*num_clu + num_pf*num_pf
                
                edge_index = np.zeros((2, num_edges))
                edge_attr = np.zeros((num_edges, self.num_edge_features))
                
                n = 0
                
                # Construct edges for clu clusters
                
                for i in range(num_clu):
                    for j in range(num_clu):
                        
                        dR_ij = (clu_eta[i]-clu_eta[j])**2 + (clu_phi[i]-clu_phi[j])**2
                        dEta_ij = clu_eta[i]-clu_eta[j]
                        dPhi_ij = clu_phi[i]-clu_phi[j]
                        
                        kt2_i = (clu_e[i]/np.cosh(clu_eta[i]))**2
                        kt2_j = (clu_e[j]/np.cosh(clu_eta[j]))**2
                        
                        edge_index[0,n] = i
                        edge_index[1,n] = j
                        
                        # clu euclidian
                        
                        edge_attr[n,0] = dR_ij
                        edge_attr[n,1] = dEta_ij
                        edge_attr[n,2] = dPhi_ij
                        
                        # clu tk metric
                        
                        edge_attr[n,3] = ktmetric(kt2_i, kt2_j, dR_ij, -1)
                        edge_attr[n,4] = ktmetric(kt2_i, kt2_j, dR_ij,  0)
                        edge_attr[n,5] = ktmetric(kt2_i, kt2_j, dR_ij,  1)
                        
                        # Euclidian distance attributes for pf are left empty
                        
                        edge_attr[n,6] = 0.0
                        edge_attr[n,7] = 0.0
                        edge_attr[n,8] = 0.0
                        
                        # pf metric is empty here
                        
                        edge_attr[n,9] = 0.0
                        edge_attr[n,10] = 0.0
                        edge_attr[n,11] = 0.0
                        
                        n += 1
                        
                # Construct edges for pf clusters
                
                for i in range(num_pf):
                    for j in range(num_pf):
                        
                        dR_ij = (pf_eta[i]-pf_eta[j])**2 + (pf_phi[i]-pf_phi[j])**2
                        dEta_ij = pf_eta[i]-pf_eta[j]
                        dPhi_ij = pf_phi[i]-pf_phi[j]
                        
                        kt2_i = (pf_p[i]/np.cosh(pf_eta[i]))**2
                        kt2_j = (pf_p[j]/np.cosh(pf_eta[j]))**2
                        
                        edge_index[0,n] = i + num_clu
                        edge_index[1,n] = j + num_clu
                        
                        # clu euclidian is zero here
                        
                        edge_attr[n,0] = 0.0
                        edge_attr[n,1] = 0.0
                        edge_attr[n,2] = 0.0
                        
                        # clu tk metric is zero here
                        
                        edge_attr[n,3] = 0.0
                        edge_attr[n,4] = 0.0
                        edge_attr[n,5] = 0.0
                        
                        # pf euclidian metrics
                        
                        edge_attr[n,6] = dR_ij
                        edge_attr[n,7] = dEta_ij
                        edge_attr[n,8] = dPhi_ij
                        
                        # pf tk metric
                        
                        edge_attr[n, 9] = ktmetric(kt2_i, kt2_j, dR_ij, -1)
                        edge_attr[n,10] = ktmetric(kt2_i, kt2_j, dR_ij,  0)
                        edge_attr[n,11] = ktmetric(kt2_i, kt2_j, dR_ij,  1)
                        
                        n += 1
                
                for i in range(num_nodes):
                    y[i] = self.is_e[event]
                
                x = torch.FloatTensor(x)
                y = torch.LongTensor(y)
                u = torch.FloatTensor(u)
                
                edge_index = torch.LongTensor(edge_index)
                edge_attr = torch.FloatTensor(edge_attr)
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, u=u)
                dataset.append(data)
        
        return dataset
