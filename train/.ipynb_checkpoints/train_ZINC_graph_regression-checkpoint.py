"""
    Utility function for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

from train.metrics import MAE

def train_epoch(model, optimizer, device, data_loader, epoch, LPE):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):

        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)
        batch_e = batch_graphs.edata['feat'].to(device)

        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()  
        
        if LPE == 'node':
            batch_EigVecs = batch_graphs.ndata['EigVecs'].to(device)
            #random sign flipping
            sign_flip = torch.rand(batch_EigVecs.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            
            batch_EigVals = batch_graphs.ndata['EigVals'].to(device)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_EigVecs, batch_EigVals)

        elif LPE == 'edge':
            batch_diff = batch_graphs.edata['diff'].to(device)
            batch_prod = batch_graphs.edata['product'].to(device)
            batch_EigVals = batch_graphs.edata['EigVals'].to(device)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_diff, batch_prod, batch_EigVals)
            
        else:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer

def evaluate_network(model, device, data_loader, epoch, LPE):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0

    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)

            if LPE == 'node':
                batch_EigVecs = batch_graphs.ndata['EigVecs'].to(device)
                batch_EigVals = batch_graphs.ndata['EigVals'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_EigVecs, batch_EigVals)

            elif LPE == 'edge':
                batch_diff = batch_graphs.edata['diff'].to(device)
                batch_prod = batch_graphs.edata['product'].to(device)
                batch_EigVals = batch_graphs.edata['EigVals'].to(device)
                batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_diff, batch_prod, batch_EigVals)
            
            else:
                batch_scores = model.forward(batch_graphs, batch_x, batch_e)
                
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)

        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae

