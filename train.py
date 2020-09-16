import models
import os
import torch
import numpy as np
import tools

from torch_geometric.data import DataLoader
from data import DataGrabber


def train(model, loader, optimiser, device, model_name,
          update_interval = 100):
    
    model = model.to(device)
    running_loss = 0
    losses = []
    epochs = []
    
    i = 0
    
    for epoch in range(num_epochs):
        for data in loader:
            data = data.to(device)
            target = data.y
            
            optimizer.zero_grad()
            loss = torch.nn.CrossEntropyLoss()
            pred = model.softpredict(data)
            output = loss(pred, target)
            output.backward()
            optimizer.step()
            
            running_loss += output.item()
            
            i += 1
            
            if i % update_interval == 0:
                
                print(f'epoch {epoch+1}: mean loss = {running_loss/i}')
                
                losses.append(running_loss/i)
                epochs.append(epoch)
                
        torch.save({'model_state_dict':model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'loss':running_loss,
                    'epoch':epoch,
                    'loss_list':losses,
                    'epoch_list':epochs
                    }, os.getcwd() + f'/saves/{model_name}_last.tar')
    
    tools.loss_curve(epochs, losses)


def test(model, loader, optimizer, device):
    
    model.to(device)
    model.eval()
    host = torch.device('cpu')
    
    pred_list = []
    target_list = []
        
    for data in loader:
        with torch.no_grad():
            
            data = data.to(device)
            target = data.y
            pred = model.softpredict(data)
                      
            pred = pred.to(host).numpy()
            target = target.to(host).numpy()
            pred = pred.transpose()[1]
            
            pred_list.append(pred)
            target_list.append(target)
    
    pred = np.concatenate(pred_list).ravel()
    target = np.concatenate(target_list).ravel()
    tools.roc_curve(target, pred)
    
##############################################################################

# ROOT file location and filename

dir_path = os.getcwd() + '/data/'
file_name = 'output.root'

batch_size = 512
num_epochs = 16
train_size = 10000
test_size  = 10000

model = None
model_name = 'dcnet'

##############################################################################

# Datagrabber is just an interface for accessing entries from ROOT file

datagrabber = DataGrabber(dir_path, file_name)

train_dataset = datagrabber.get_dataset(0, train_size)
test_dataset = datagrabber.get_dataset(train_size, train_size + test_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

D = datagrabber.num_node_features
E = datagrabber.num_edge_features
C = 2

if model_name == 'gcnet':
    model = models.GCNet(D, C, 64)
    
elif model_name == 'gatnet':
    model = models.GATNet(D, C)
    
elif model_name == 'dcnet':
    model = models.DECNet(D, C, k=4)
    
elif model_name == 'nnnet':
    model = models.NNNet(D, C, E=E)
    
elif model_name == 'splinenet':
    model = models.SplineNet(D, C)
    
elif model_name == 'sagenet':
    model = models.SAGENet(D, C)
    
elif model_name == 'sgnet':
    model = models.SGNet(D, C)
    
elif model_name == 'ginenet':
    model = models.GINENet(D, C)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
device = torch.device('cuda')

train(model, train_loader, optimizer, device, model_name)
test(model, test_loader, optimizer, device)




