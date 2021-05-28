"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

from ogb.graphproppred import Evaluator

def train_epoch(model, optimizer, device, data_loader, epoch, LPE):
    model.train()
    evaluator = Evaluator(name = "ogbg-molhiv")
    
    epoch_loss = 0
    epoch_train_auc = 0

    targets=torch.tensor([]).to(device)
    scores=torch.tensor([]).to(device)
    
    for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
        
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        
        batch_targets = batch_targets.to(device)
        optimizer.zero_grad()  
        
        if LPE == 'node':
            
            batch_EigVecs = batch_graphs.ndata['EigVecs'].to(device)
            #random sign flipping
            sign_flip = torch.rand(batch_EigVecs.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            batch_EigVecs = batch_EigVecs * sign_flip.unsqueeze(0)
            
            batch_EigVals = batch_graphs.ndata['EigVals'].to(device)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_EigVecs, batch_EigVals)

        elif LPE == 'edge':
            
            batch_diff = batch_graphs.edata['diff'].to(device)
            batch_prod = batch_graphs.edata['product'].to(device)
            batch_EigVals = batch_graphs.edata['EigVals'].to(device)
            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_diff, batch_prod, batch_EigVals)
            
        else:
            batch_scores = model.forward(batch_graphs, batch_x, batch_e)
            

        targets = torch.cat((targets, batch_targets), 0)
        scores = torch.cat((scores, batch_scores), 0)
        
        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        
    
    input_dict = {"y_true": targets, "y_pred": scores}
    epoch_train_auc = evaluator.eval(input_dict)['rocauc']  

    epoch_loss /= (iter + 1)
    
    return epoch_loss, epoch_train_auc, optimizer

def evaluate_network(model, device, data_loader, epoch, LPE):
    model.eval()
    evaluator = Evaluator(name = "ogbg-molhiv")
    
    epoch_test_loss = 0
    epoch_test_auc = 0
    
    targets=torch.tensor([]).to(device)
    scores=torch.tensor([]).to(device)
    
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
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
                

            targets = torch.cat((targets, batch_targets), 0)
            scores = torch.cat((scores, batch_scores), 0)         
            
            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()

            
    input_dict = {"y_true": targets, "y_pred": scores}
    epoch_test_auc = evaluator.eval(input_dict)['rocauc']
            
    epoch_test_loss /= (iter + 1)

    return epoch_test_loss, epoch_test_auc

