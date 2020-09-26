import numpy as np
import uproot
import os
import networkx as nx
import matplotlib.pyplot as plt
import tools

from numba import jit

rc = {'font.size': 18, 
      'mathtext.fontset': 'cm',
      'legend.fontsize': 24}
plt.rcParams.update(rc)


@jit(nopython = True)
def filter_clusters(ref_pt, clu_n, is_e, n_min, pt_min):
    num_t_ref = 0
    num_f_ref = 0
    
    num_t_clu = 0
    num_f_clu = 0
    
    for i in range(len(is_e)):
        n = clu_n[i]
        if ref_pt[i] > pt_min and n > n_min:
            if is_e[i]:
                num_t_ref += 1
                num_t_clu += n
            else:
                num_f_ref += 1
                num_f_clu += n
            
    t_idxs = np.zeros(num_t_ref, dtype=np.int32)
    f_idxs = np.zeros(num_f_ref, dtype=np.int32)
    
    t_i = 0
    f_i = 0
    
    for i in range(len(is_e)):
        if ref_pt[i] > pt_min and clu_n[i] > n_min:
            if is_e[i]:
                t_idxs[t_i] = i
                t_i += 1
            else:
                f_idxs[f_i] = i
                f_i += 1
    
    return num_t_ref, num_f_ref, t_idxs, f_idxs, num_t_clu, num_f_clu


def plot_clusters(path, connect='tree', n_min=2, pt_min=200.0):
    events = uproot.open(path)['ntuplizer']['tree']
    
    fig = plt.figure()
    fig.set_size_inches(8, 10)
    
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    
    ax1.set_ylim((-3.5, 3.5))
    ax1.set_xlim((-1.6, 1.6))
    ax2.set_ylim((-3.5, 3.5))
    ax2.set_xlim((-1.6, 1.6))
    
    ax1.set_facecolor('black')
    ax2.set_facecolor('black')
    
    ax1.set_title(r'Signal $n_\mathrm{min}=$' + f'{n_min},' + 
                  r'$p_\mathrm{T, min}=$' + f'{pt_min/10} GeV')
    ax2.set_title(r'Background $n_\mathrm{min}=$' + f'{n_min},' + 
                  r'$p_\mathrm{T, min}=$' + f'{pt_min/10} GeV')
    
    ax1.set_ylabel(r'$\phi$')
    ax2.set_ylabel(r'$\phi$')
    ax2.set_xlabel(r'$\eta$')
    
    ax1.axhline(+np.pi, color = 'white', linestyle = '--')
    ax1.axhline(-np.pi, color = 'white', linestyle = '--')
    ax2.axhline(+np.pi, color = 'white', linestyle = '--')
    ax2.axhline(-np.pi, color = 'white', linestyle = '--')
    
    ax1.axvline(+1.5, color = 'white', linestyle = '--')
    ax1.axvline(-1.5, color = 'white', linestyle = '--')
    ax2.axvline(+1.5, color = 'white', linestyle = '--')
    ax2.axvline(-1.5, color = 'white', linestyle = '--')
    
    is_e = events['is_e'].array()
    ref_eta = events['image_gsf_ref_eta'].array()
    ref_phi = events['image_gsf_ref_phi'].array()
    gsf_ref_p = events['image_gsf_ref_p'].array()
    gsf_ref_pt = events['image_gsf_ref_pt'].array()
    clu_eta = events['image_clu_eta'].array()
    clu_phi = events['image_clu_phi'].array()
    clu_e = events['image_clu_e'].array()
    clu_n = events['image_clu_n'].array()
    
    num_t_ref, num_f_ref, t_idxs, f_idxs, num_t_clu, num_f_clu = filter_clusters(gsf_ref_pt, clu_n, is_e, n_min, pt_min)
    
    print(f'{num_t_ref} electron superclusters, {num_t_clu} electron clusters')
    print(f'{num_f_ref} electron superclusters, {num_f_clu} electron clusters')
    
    etas_t, etas_f = np.zeros(num_t_clu), np.zeros(num_f_clu)
    phis_t, phis_f = np.zeros(num_t_clu), np.zeros(num_f_clu)
    engy_t, engy_f = np.zeros(num_t_clu), np.zeros(num_f_clu)
    
    t_i = 0
    f_i = 0
    
    for i in range(num_t_ref):
        idx = int(t_idxs[i])
        n = clu_n[idx]
        
        for j in range(n):
            etas_t[t_i] = ref_eta[idx] + clu_eta[idx][j]
            phis_t[t_i] = ref_phi[idx] + clu_phi[idx][j]
            engy_t[t_i] = clu_e[idx][j]
            t_i += 1
    
    for i in range(num_f_ref):
        idx = int(f_idxs[i])
        n = clu_n[idx]
        
        for j in range(n):
            etas_f[f_i] = ref_eta[idx] + clu_eta[idx][j]
            phis_f[f_i] = ref_phi[idx] + clu_phi[idx][j]
            engy_f[f_i] = clu_e[idx][j]
            f_i += 1
    
    if connect == 'tree':
        for i in range(num_t_ref):
            idx = t_idxs[i]
            eta_ref = ref_eta[idx]
            phi_ref = ref_phi[idx]
            n = clu_n[idx]
            
            for j in range(n):
                y = [phi_ref, phi_ref + clu_phi[idx][j]]
                x = [eta_ref, eta_ref + clu_eta[idx][j]]
                ax1.plot(x, y, color = 'white', lw = 0.5)
                
        
        for i in range(num_f_ref):
            idx = f_idxs[i]
            eta_ref = ref_eta[idx]
            phi_ref = ref_phi[idx]
            n = clu_n[idx]
            
            for j in range(n):
                y = [phi_ref, phi_ref + clu_phi[idx][j]]
                x = [eta_ref, eta_ref + clu_eta[idx][j]]
                ax2.plot(x, y, color = 'white', lw = 0.5)

    elif connect == 'full':
        for i in range(num_t_ref):
            idx = t_idxs[i]
            n = clu_n[idx]
            
            eta = clu_eta[idx]
            phi = clu_phi[idx]
            
            eta_ref = ref_eta[idx]
            phi_ref = ref_phi[idx]
            
            coords = np.transpose(np.array([eta, phi]))
            D = tools.l2_norm(coords, coords)
            G = nx.from_numpy_matrix(D)
            #G = nx.minimum_spanning_tree(G)
            
            for k in G.edges():
                v0, v1 = k
                y = [phi_ref + phi[v0], phi_ref + phi[v1]]
                x = [eta_ref + eta[v0], eta_ref + eta[v1]]
                ax1.plot(x, y, color = 'white', lw = 0.5)

        for i in range(num_f_ref):
            idx = f_idxs[i]
            n = clu_n[idx]
            
            eta = clu_eta[idx]
            phi = clu_phi[idx]
            
            eta_ref = ref_eta[idx]
            phi_ref = ref_phi[idx]
            
            coords = np.transpose(np.array([eta, phi]))
            D = tools.l2_norm(coords, coords)
            G = nx.from_numpy_matrix(D)
            #G = nx.minimum_spanning_tree(G)
            
            for k in G.edges():
                v0, v1 = k
                y = [phi_ref + phi[v0], phi_ref + phi[v1]]
                x = [eta_ref + eta[v0], eta_ref + eta[v1]]
                ax2.plot(x, y, color = 'white', lw = 0.5)
            
    
    
    ax1.scatter(etas_t, phis_t, s = 4.0, c=engy_t,
    norm = plt.Normalize(vmin=0, vmax=1),
    cmap = "plasma", marker = 's')

    ax2.scatter(etas_f, phis_f, s = 4.0, c=engy_f,
    norm = plt.Normalize(vmin=0, vmax=1),
    cmap = "plasma", marker = 's')


def plot_supercluster_graph(path, n):
    events = uproot.open(path)['ntuplizer']['tree']
    
    ref_eta = events['image_gsf_ref_eta'].array()
    ref_phi = events['image_gsf_ref_phi'].array()
    clu_eta = events['image_clu_eta'].array()
    clu_phi = events['image_clu_phi'].array()
    clu_e = events['image_clu_e'].array()
    clu_n = events['image_clu_n'].array()
    
        
    eta = None
    phi = None
    
    for i in range(events.numentries):
        if clu_n[i] == n:
            eta = clu_eta[i]
            phi = clu_phi[i]
            e = clu_e[i]
            break
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    axes = [ax1, ax2, ax3, ax4]
    
    for ax in axes:
        ax.set_facecolor('black')
    
    coords = np.transpose(np.array([eta, phi]))
    D = tools.l2_norm(coords, coords)
    G = nx.from_numpy_matrix(D)
    
    for k in G.edges():
        v0, v1 = k
        y = [phi[v0], phi[v1]]
        x = [eta[v0], eta[v1]]
        ax1.plot(x, y, color = 'white', lw = 1.0)
        
    G = nx.minimum_spanning_tree(G)
    
    for k in G.edges():
        v0, v1 = k
        y = [phi[v0], phi[v1]]
        x = [eta[v0], eta[v1]]
        ax2.plot(x, y, color = 'white', lw = 1.0)
    
    for k in range(n):
        y = [0, phi[k]]
        x = [0, eta[k]]
        ax3.plot(x, y, color = 'white', lw = 1.0)

    D_k_nn = tools.k_nn_graph(D, 3)
    G = nx.from_numpy_matrix(D_k_nn)
    
    for k in G.edges():
        v0, v1 = k
        y = [phi[v0], phi[v1]]
        x = [eta[v0], eta[v1]]
        ax4.plot(x, y, color = 'white', lw = 1.0)
    
    for ax in axes:
        ax.scatter(eta, phi, s = 64.0, c=e,
        norm = plt.Normalize(vmin=0, vmax=1),
        cmap = "plasma", marker = 's', zorder = 10)
        

def plot_avgs(path):
    events = uproot.open(path)['ntuplizer']['tree']
    
    clu_eta = events['image_clu_eta'].array()
    clu_phi = events['image_clu_phi'].array()
    clu_e = events['image_clu_e'].array()
    clu_n = events['image_clu_n'].array()
    
    pf_eta = events['image_pf_eta'].array()
    pf_phi = events['image_pf_phi'].array()
    pf_p = events['image_pf_p'].array()
    pf_n = events['image_pf_n'].array()
    
    is_e = events['is_e'].array()
    
    c_eta_max, c_eta_min, c_num_eta = 1.0, -1.0, 1000
    c_phi_max, c_phi_min, c_num_phi = 3.2, -3.2, 1000
    
    p_eta_max, p_eta_min, p_num_eta = 0.001, -0.001, 100
    p_phi_max, p_phi_min, p_num_phi = 1.0, -1.0, 100
    
    c_deta = (c_eta_max-c_eta_min)/c_num_eta
    c_dphi = (c_phi_max-c_phi_min)/c_num_phi
    
    p_deta = (p_eta_max-p_eta_min)/p_num_eta
    p_dphi = (p_phi_max-p_phi_min)/p_num_phi
    
    cp_grid_e = np.zeros((c_num_eta, c_num_phi))
    cp_grid_n = np.zeros((c_num_eta, c_num_phi))
    
    cn_grid_e = np.zeros((c_num_eta, c_num_phi))
    cn_grid_n = np.zeros((c_num_eta, c_num_phi))
    
    pp_grid_e = np.zeros((p_num_eta, p_num_phi))
    pp_grid_n = np.zeros((p_num_eta, p_num_phi))
    
    pn_grid_e = np.zeros((p_num_eta, p_num_phi))
    pn_grid_n = np.zeros((p_num_eta, p_num_phi))
    
    for example in range(100000):
        n = clu_n[example]
        e = clu_e[example]
        eta = clu_eta[example]
        phi = clu_phi[example]
        is_example_e = is_e[example]
        
        if n > 0:
            if np.all(np.abs(eta) < c_eta_max):
        
                for k in range(n):
                    
                    i = int(np.floor((eta[k] + c_eta_max)/c_deta))
                    j = int(np.floor((phi[k] + c_phi_max)/c_dphi))
                    
                    if is_example_e:
                        cp_grid_e[i,j] += e[k]
                        cp_grid_n[i,j] += 1
                        
                    else:
                        cn_grid_e[i,j] += e[k]
                        cn_grid_n[i,j] += 1
    
    for example in range(100000):
        n = pf_n[example]
        p = pf_p[example]
        eta = pf_eta[example]
        phi = pf_phi[example]
        is_example_e = is_e[example]
        
        if n > 0:
            if np.all(np.abs(eta) < p_eta_max):
        
                for k in range(n):
                    
                    i = int(np.floor((eta[k] + p_eta_max)/p_deta))
                    j = int(np.floor((phi[k] + p_phi_max)/p_dphi))
                    
                    if is_example_e:
                        pp_grid_e[i,j] += p[k]
                        pp_grid_n[i,j] += 1
                        
                    else:
                        pn_grid_e[i,j] += p[k]
                        pn_grid_n[i,j] += 1
                    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    
    ax1.set_xlim(400, 550)
    ax3.set_xlim(400, 550)
    
    ax1.set_ylim(450, 550)
    ax3.set_ylim(450, 550)
    
    
    ax1.imshow(cp_grid_e, interpolation='nearest')
    ax3.imshow(cn_grid_e, interpolation='nearest')
    ax2.imshow(pp_grid_n, interpolation='nearest')
    ax4.imshow(pn_grid_n, interpolation='nearest')
    
        
def plot_stats(path):
    events = uproot.open(path)['ntuplizer']['tree']
    
    events.show()
    
    print('')
    print('ECAL cluster sizes:')
    print('')
    
    clu_n = events['image_clu_n'].array()
    pf_n = events['image_pf_n'].array()
    pf_pdgid = events['image_pf_pdgid'].array().flatten()
    
    clu_num = np.size(clu_n)
    pf_num = np.size(pf_n)
    pf_id_num = np.size(pf_pdgid)
    
    for i in range(20):
        clu_count = np.count_nonzero(clu_n == i)
        print(f'n = {i}: {clu_count} ({round(clu_count/clu_num, 3)})')
    
    print('')
    print('PF cluster sizes:')
    print('')
        
    for i in range(22):
        pf_count = np.count_nonzero(pf_n == i)
        print(f'n = {i}: {pf_count} ({round(pf_count/pf_num, 3)})')
    
    print('')
    print('PF PDGID distribution:')
    print('')
        
    for i in range(300):
        pf_id_count = np.count_nonzero(pf_pdgid == i)
        if pf_id_count > 0:
            print(f'id = {i}: {pf_id_count} ({round(pf_id_count/pf_id_num, 3)})')




if __name__ == "__main__" :
    root_path = os.getcwd() + '/data/output.root'
    #plot_clusters(root_path, connect='tree')
    #plot_supercluster_graph(root_path, 16)
    #plot_stats(root_path)
    plot_avgs(root_path)


