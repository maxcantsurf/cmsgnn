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
def calc_graph(num_nodes, num_edges, num_edge_features, eta, phi):
    n = 0
    
    edge_index = np.zeros((2, num_edges))
    edge_attr = np.zeros((num_edges, num_edge_features))
                
    for i in range(num_nodes):
        for j in range(num_nodes):
            
            dR_ij = (eta[i]-eta[j])**2 + (phi[i]-phi[j])**2
            dEta_ij = eta[i]-eta[j]
            dPhi_ij = phi[i]-phi[j]
            
            edge_index[0,n] = i
            edge_index[1,n] = j
            
            edge_attr[n,0] = dR_ij
            edge_attr[n,1] = dEta_ij
            edge_attr[n,2] = dPhi_ij
            
            n += 1
    
    return edge_index, edge_attr


def ktmetric(kt2_i, kt2_j, dR2_ij, p = -1, R = 1.0):
    """
    kt-algorithm type distance measure.
    
    Args:
        kt2_i      : Particle 1 pt squared
        kt2_j      : Particle 2 pt squared
        delta2_ij  : Angular seperation between particles squared (deta**2 + dphi**2)
        R          : Radius parameter
        
        p =  1     : (p=1) kt-like, (p=0) Cambridge/Aachen, (p=-1) anti-kt like
    Returns:
        distance measure
    """

    return np.min([kt2_i**(2*p), kt2_j**(2*p)]) * (dR2_ij/R**2)


class DataGrabber():
    
    def __init__(self, dir_path, file_name):
        self.dir_path = dir_path
        self.file_name = file_name
        self.root_file = uproot.open(dir_path + file_name)
        self.events_full = self.root_file['ntuplizer']['tree']
        self.numentries = self.events_full.numentries
        
        with open(os.getcwd() + '/dataconfig.yml') as file:
            dataconfig = yaml.full_load(file)
        
        self.event_feature_names = dataconfig['event_feature_names']
        self.cluster_feature_names = dataconfig['cluster_feature_names']
        self.pfcluster_feature_names = dataconfig['pfcluster_feature_names']
        
        self.num_node_features = 8
        self.num_edge_features = 3
        

    def get_dataset(self, entrystart, entrystop):
        
        self.event_features = {}
        self.cluster_features = {}
        self.pfcluster_features = {}
        
        for feature in self.event_feature_names:
            self.event_features[feature] = self.events_full[feature].array(entrystart=entrystart, entrystop=entrystop)
        
        for feature in self.cluster_feature_names:
            self.cluster_features[feature] = self.events_full[feature].array(entrystart=entrystart, entrystop=entrystop)
        
        for feature in self.pfcluster_feature_names:
            self.pfcluster_features[feature] = self.events_full[feature].array(entrystart=entrystart, entrystop=entrystop)
        
        self.is_e = self.events_full['is_e'].array(entrystart=entrystart, entrystop=entrystop)
        
        dataset = []
        
        for event in tqdm(range(len(self.is_e))):
            
            num_clu = self.event_features['image_clu_n'][event]
            num_pf = self.event_features['image_pf_n'][event]
            
            num_nodes = num_clu + num_pf
            
            # Discard events which have one cluster or less
            
            if num_nodes > 0:
                
                clu_eta = self.cluster_features['image_clu_eta'][event]
                clu_phi = self.cluster_features['image_clu_phi'][event]
                
                clu_e = self.cluster_features['image_clu_e'][event]
                clu_nhit = self.cluster_features['image_clu_nhit'][event]
                
                pf_eta = self.pfcluster_features['image_pf_eta'][event]
                pf_phi = self.pfcluster_features['image_pf_phi'][event]
                
                pf_p = self.pfcluster_features['image_pf_p'][event]
                pf_pdgid = self.pfcluster_features['image_pf_pdgid'][event]
                
                
                num_edge_features = self.num_edge_features
                num_edges = num_nodes*num_nodes
                
                x = np.zeros((num_nodes, self.num_node_features))
                y = np.zeros(num_nodes)
                u = np.zeros((num_nodes, 0))
                
                
                n = 0
                
                for i in range(num_clu):
                    x[n][0] = clu_eta[i]
                    x[n][1] = clu_phi[i]
                    x[n][2] = clu_e[i]
                    x[n][3] = clu_nhit[i]
                    x[n][4] = 0.0
                    x[n][5] = 0.0
                    x[n][6] = 1.0
                    x[n][7] = 0.0
                    
                    n += 1
                
                for i in range(num_pf):
                    x[n][0] = pf_eta[i]
                    x[n][1] = pf_phi[i]
                    x[n][2] = 0.0
                    x[n][3] = 0.0
                    x[n][4] = pf_p[i]
                    x[n][5] = pf_pdgid[i]
                    x[n][6] = 0.0
                    x[n][7] = 1.0
                    
                    n += 1
                
                eta = np.concatenate((clu_eta,  pf_eta))
                phi = np.concatenate((clu_phi,  pf_phi))
                
                n = 0
    
                edge_index = np.zeros((2, num_edges))
                edge_attr = np.zeros((num_edges, num_edge_features))
                            
                for i in range(num_nodes):
                    for j in range(num_nodes):
                        
                        dR_ij = (eta[i]-eta[j])**2 + (phi[i]-phi[j])**2
                        dEta_ij = eta[i]-eta[j]
                        dPhi_ij = phi[i]-phi[j]
                        
                        edge_index[0,n] = i
                        edge_index[1,n] = j
                        
                        edge_attr[n,0] = 1.0
                        edge_attr[n,1] = 1.0
                        edge_attr[n,2] = 1.0
                        
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
