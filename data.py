import numpy as np
import uproot
import torch
import yaml
import os
import aux

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


@jit(nopython = True)
def pdgid_to_onehot(pdgid):
    if pdgid == 211:
        return 0
    elif pdgid == 22:
        return 1
    elif pdgid == 130:
        return 2
    elif pdgid == 11:
        return 3
    elif pdgid == 13:
        return 4
    elif pdgid == 2:
        return 5
    elif pdgid == 1:
        return 6
    else:
        return 7


@jit(nopython = True)
def make_nodes_both(clu_eta, clu_phi, clu_e, clu_nhit, num_clu,
                    pf_eta, pf_phi, pf_p, pf_pdgid, num_pf,
                    num_node_features, num_global_features):
    
    num_nodes = num_clu + num_pf
    x = np.zeros((num_nodes, num_node_features))
    n = 0
    
    # TODO: Make this cleaner
    # clu node attributes
    
    for i in range(num_clu):
        x[n][0] = clu_eta[i]
        x[n][1] = clu_phi[i]
        x[n][2] = clu_e[i]
        x[n][3] = clu_nhit[i]
        
        n += 1
        
    # pf node attributes
    
    for i in range(num_pf):
        
        x[n][4] = pf_eta[i]
        x[n][5] = pf_phi[i]
        x[n][6] = pf_p[i]
        
        index = pdgid_to_onehot(pf_pdgid[i])
        
        x[n][7+index] = 1.0
        
        n += 1
        
        
    
    return x


@jit(nopython = True)
def make_globals_both(num_clu, num_pf, num_global_features):
    u = np.zeros(num_global_features)
    u[0] = num_clu
    u[1] = num_pf
    
    return u


@jit(nopython = True)
def make_edges_both(clu_eta, clu_phi, clu_e, num_clu, 
                     pf_eta, pf_phi, pf_p, num_pf, num_edge_features):
    
    num_edges = num_clu*num_clu + num_pf*num_pf
                
    edge_index = np.zeros((2, num_edges))
    edge_attr = np.zeros((num_edges, num_edge_features))
    
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
            
            # clu tk metric
            
            edge_attr[n,0] = ktmetric(kt2_i, kt2_j, dR_ij,  0)
            
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
            
            # pf euclidian metrics
            
            # pf tk metric
            
            edge_attr[n,1] = ktmetric(kt2_i, kt2_j, dR_ij,  0)
            
            n += 1
    
    return edge_index, edge_attr


@jit(nopython = True)
def make_nodes_clu(clu_eta, clu_phi, clu_e, clu_nhit, num_clu,
                        num_node_features):
    
    x = np.zeros((num_clu+1, num_node_features))

    for i in range(1,num_clu+1):
            x[i][0] = clu_eta[i-1]
            x[i][1] = clu_phi[i-1]
            x[i][2] = clu_e[i-1]
            x[i][3] = clu_nhit[i-1]
            
    return x


@jit(nopython = True)
def make_globals_clu(num_clu, num_global_features):
    
    u = np.zeros(num_global_features)
    
    u[0] = num_clu
    
    return u
            

@jit(nopython = True)
def make_edges_clu(clu_eta, clu_phi, clu_e, num_clu, num_edge_features):
    
    num_edges = num_clu*num_clu
                
    edge_index = np.zeros((2, num_edges))
    edge_attr = np.zeros((num_edges, num_edge_features))
    
    n = 0
    
    # Construct edges for clu clusters
    
    for i in range(1,num_clu+1):
        for j in range(1,num_clu+1):
            
            dR_ij = (clu_eta[i-1]-clu_eta[j-1])**2 + (clu_phi[i-1]-clu_phi[j-1])**2
            dEta_ij = clu_eta[i-1]-clu_eta[j-1]
            dPhi_ij = clu_phi[i-1]-clu_phi[j-1]
            
            kt2_i = (clu_e[i-1]/np.cosh(clu_eta[i-1]))**2
            kt2_j = (clu_e[j-1]/np.cosh(clu_eta[j-1]))**2
            
            edge_index[0,n] = i
            edge_index[1,n] = j
            
            # clu euclidian
            
            # clu tk metric
            
            edge_attr[n,0] = ktmetric(kt2_i, kt2_j, dR_ij, -1)
            edge_attr[n,1] = ktmetric(kt2_i, kt2_j, dR_ij,  0)
            edge_attr[n,2] = ktmetric(kt2_i, kt2_j, dR_ij,  1)
            
            edge_attr[n,3] = dR_ij
            edge_attr[n,4] = dEta_ij
            edge_attr[n,5] = dPhi_ij
            
            n += 1

    return edge_index, edge_attr


def compute_reweights(pt, eta, args):
    
    pt_binedges  = np.linspace(0.0, 300.0, 1000)
    eta_binedges = np.linspace(-3.1, 3.1,   100)

    trn_weights = aux.reweightcoeff2D(pt, eta, data.trn.y, pt_binedges, eta_binedges,
        shape_reference = 'background', max_reg = 50.0)

    ### Plot some kinematic variables
    targetdir = f'./figs/eid/{args["config"]}/reweight/1D_kinematic/'
    os.makedirs(targetdir, exist_ok = True)

    tvar = ['trk_pt', 'trk_eta', 'trk_phi', 'trk_p']
    for k in tvar:
        plots.plotvar(x = data.trn.x[:, data.VARS.index(k)], y = data.trn.y, weights = trn_weights, var = k, NBINS = 70,
            targetdir = targetdir, title = 'training reweight reference: {}'.format(args['reweight_param']['mode']))

    print(__name__ + '.compute_reweights: [done]')

    return trn_weights



@jit(nopython = True)
def get_cut_indices(num_clu):
    
    num_accepted = 0
    
    for example in range(len(num_clu)):
        if num_clu[example] > 0:
            num_accepted += 1
    
    accepted_indices = np.zeros(num_accepted, dtype=np.int64)
    i = 0
    
    for example in range(len(num_clu)):
        if num_clu[example] > 0:
            accepted_indices[i] = example
            i += 1
        
    return accepted_indices





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
        
    def get_dataset(self, entrystart, entrystop, cluster='both'):
        
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
        num_clu = self.global_features['image_clu_n']
        acc_examples = get_cut_indices(num_clu)
        
        for event in tqdm(acc_examples):
            
            num_clu = self.global_features['image_clu_n'][event]
            num_pf = self.global_features['image_pf_n'][event]
            
            clu_eta = self.cluster_features['image_clu_eta'][event]
            clu_phi = self.cluster_features['image_clu_phi'][event]
            
            clu_e = self.cluster_features['image_clu_e'][event]
            clu_nhit = self.cluster_features['image_clu_nhit'][event]
            
            pf_eta = self.pfcluster_features['image_pf_eta'][event]
            pf_phi = self.pfcluster_features['image_pf_phi'][event]
            
            pf_p = self.pfcluster_features['image_pf_p'][event]
            pf_pdgid = self.pfcluster_features['image_pf_pdgid'][event]
            
            ref_eta = self.global_features['image_gsf_ref_eta'][event]
            ref_phi = self.global_features['image_gsf_ref_phi'][event]
            
            
            if (np.all(np.abs(ref_eta + clu_eta) < 1.45)):
            
                if cluster == 'BOTH':
                
                    self.num_node_features = 15
                    self.num_edge_features = 2
                    self.num_global_features = 2
                    
                    x = make_nodes_both(clu_eta, clu_phi, clu_e, clu_nhit, num_clu,
                        pf_eta, pf_phi, pf_p, pf_pdgid, num_pf,
                        self.num_node_features, self.num_global_features)
                    
                    u = make_globals_both(num_clu, num_pf, self.num_global_features)
                    
                    edge_index, edge_attr = make_edges_both(clu_eta, clu_phi,
                                                            clu_e, num_clu, 
                                                            pf_eta, pf_phi, 
                                                            pf_p, num_pf, 
                                                            self.num_edge_features)
                    
                elif cluster == 'ECAL':
                    
                    self.num_node_features = 4
                    self.num_edge_features = 3
                    self.num_global_features = 1
                    
                    x = make_nodes_clu(clu_eta, clu_phi, clu_e, clu_nhit, num_clu,
                                               self.num_node_features)
                    
                    u = make_globals_clu(num_clu, self.num_global_features)
                    
                    edge_index, edge_attr = make_edges_clu(clu_eta, clu_phi,
                                                                clu_e, num_clu,
                                                                self.num_edge_features)
                    
                else:
                    print('Cluster type must be either "ECAL" or "BOTH"')
                
                y = np.zeros(1)
                
                y[0] = self.is_e[event]
                
                x = torch.FloatTensor(x)
                y = torch.LongTensor(y)
                u = torch.FloatTensor(u)
                
                edge_index = torch.LongTensor(edge_index)
                edge_attr = torch.FloatTensor(edge_attr)
                
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, u=u)
                dataset.append(data)
        
        return dataset
    
