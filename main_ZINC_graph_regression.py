"""
    IMPORTING LIBS
"""
import dgl

import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm



"""
    IMPORTING CUSTOM MODULES/METHODS
"""
from nets.ZINC_graph_regression.load_net import gnn_model 
from data.data import LoadData 



"""
    GPU Setup
"""
def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device



"""
    VIEWING ENCODING TYPE AND NUM PARAMS
"""
def view_model_param(LPE, net_params):
    model = gnn_model(LPE, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
        
    if LPE == 'edge':
        print('Encoding Type/Total parameters:', 'Edge Laplace Encoding/', total_param)
    elif LPE == 'node':
        print('Encoding Type/Total parameters:', 'Node Laplace Encoding', total_param)
    else:
        print('Encoding Type/Total parameters:', 'None', total_param)
    return total_param


"""
    TRAINING CODE
"""

def train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    start0 = time.time()
    per_epoch_time = []
        
    DATASET_NAME = dataset.name
    
    if net_params['LPE'] in ['edge', 'node']:
        st = time.time()
        print("[!] Computing Laplace Decompositions..")
        dataset._laplace_decomp(net_params['m'])
        print('Time taken to decompose Laplacians: ',time.time()-st)
        
    if net_params['full_graph']:
        st = time.time()
        print("[!] Adding full graph connectivity..")
        dataset._make_full_graph()
        print('Time taken to add full graph connectivity: ',time.time()-st)
        
    if net_params['LPE'] == 'edge':
        st = time.time()
        print("[!] Computing edge Laplace features..")
        dataset._add_edge_laplace_feats()
        print('Time taken to compute edge Laplace features: ',time.time()-st)
    
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    
    net_params['total_param'] = view_model_param(net_params['LPE'], net_params)
        
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write the network and optimization hyper-parameters in folder config/
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    model = gnn_model(net_params['LPE'], net_params)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params['lr_reduce_factor'],
                                                     patience=params['lr_schedule_patience'],
                                                     verbose=True)
    
    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_MAEs, epoch_val_MAEs, epoch_test_MAEs = [], [], []
    
        

    # import train and evaluate functions
    from train.train_ZINC_graph_regression import train_epoch, evaluate_network

    
    train_loader = DataLoader(trainset, num_workers=4, batch_size=params['batch_size'], shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(valset, num_workers=4, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    test_loader = DataLoader(testset, num_workers=4, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)
    
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()

                epoch_train_loss, epoch_train_mae, optimizer = train_epoch(model, optimizer, device, train_loader, epoch, net_params['LPE'])
                    
                epoch_val_loss, epoch_val_mae = evaluate_network(model, device, val_loader, epoch, net_params['LPE'])
                _, epoch_test_mae = evaluate_network(model, device, test_loader, epoch, net_params['LPE'])
                
                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                epoch_train_MAEs.append(epoch_train_mae)
                epoch_val_MAEs.append(epoch_val_mae)
                
                epoch_test_MAEs.append(epoch_test_mae)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('train/_mae', epoch_train_mae, epoch)
                writer.add_scalar('val/_mae', epoch_val_mae, epoch)
                writer.add_scalar('test/_mae', epoch_test_mae, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                        
                t.set_postfix(time=time.time()-start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_MAE=epoch_train_mae, val_MAE=epoch_val_mae,
                              test_MAE=epoch_test_mae)


                per_epoch_time.append(time.time()-start)

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, "RUN_")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(ckpt_dir + "/epoch_" + str(epoch)))

                files = glob.glob(ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch-1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break
                
                # Stop training after params['max_time'] hours
                if time.time()-start0 > params['max_time']*3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(params['max_time']))
                    break
                
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')
        
    #Return test and train metrics at best val metric
    index = epoch_val_MAEs.index(min(epoch_val_MAEs))
    
    test_mae = epoch_test_MAEs[index]
    train_mae = epoch_train_MAEs[index]
    
    print("Test MAE: {:.4f}".format(test_mae))
    print("Train MAE: {:.4f}".format(train_mae))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-start0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    with open(write_file_name + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n{}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST MAE: {:.4f}\nTRAIN MAE: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""\
          .format(DATASET_NAME, MODEL_NAME, params, net_params, model, net_params['total_param'],
                  test_mae, train_mae, epoch, (time.time()-start0)/3600, np.mean(per_epoch_time)))
        




def main():    
    """
        USER CONTROLS
    """
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help="Please give a config.json file with training/model/data/param details")
    parser.add_argument('--gpu_id', help="Please give a value for gpu id")
    parser.add_argument('--model', help="Please give a value for model name")
    parser.add_argument('--dataset', help="Please give a value for dataset name")
    parser.add_argument('--out_dir', help="Please give a value for out_dir")
    parser.add_argument('--seed', help="Please give a value for seed")
    parser.add_argument('--epochs', help="Please give a value for epochs")
    parser.add_argument('--batch_size', help="Please give a value for batch_size")
    parser.add_argument('--init_lr', help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', help="Please give a value for weight_decay")
    parser.add_argument('--print_epoch_interval', help="Please give a value for print_epoch_interval")
    parser.add_argument('--max_time', help="Please give a value for max_time")
    
    #Model details
    parser.add_argument('--full_graph', help="Please give a value for full_graph")    
    parser.add_argument('--gamma', help="Please give a value for gamma")
    parser.add_argument('--m', help="Please give a value for m")
    
    parser.add_argument('--LPE', help="Please give a value for LPE")
    parser.add_argument('--LPE_layers', help="Please give a value for LPE_layers")
    parser.add_argument('--LPE_dim', help="Please give a value for LPE_dim")
    parser.add_argument('--LPE_n_heads', help="Please give a value for LPE_n_heads")
    
    parser.add_argument('--GT_layers', help="Please give a value for GT_layers")
    parser.add_argument('--GT_hidden_dim', help="Please give a value for GT_hidden_dim")
    parser.add_argument('--GT_out_dim', help="Please give a value for GT_out_dim")
    parser.add_argument('--GT_n_heads', help="Please give a value for GT_n_heads")
    
    parser.add_argument('--residual', help="Please give a value for readout")
    parser.add_argument('--readout', help="Please give a value for readout")
    parser.add_argument('--in_feat_dropout', help="Please give a value for in_feat_dropout")
    parser.add_argument('--dropout', help="Please give a value for dropout")
    parser.add_argument('--layer_norm', help="Please give a value for layer_norm")
    parser.add_argument('--batch_norm', help="Please give a value for batch_norm")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = json.load(f)
        
    # device
    if args.gpu_id is not None:
        config['gpu']['id'] = int(args.gpu_id)
        config['gpu']['use'] = True
    device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
    # model, dataset, out_dir
    if args.model is not None:
        MODEL_NAME = args.model
    else:
        MODEL_NAME = config['model']
    if args.dataset is not None:
        DATASET_NAME = args.dataset
    else:
        DATASET_NAME = config['dataset']
    dataset = LoadData(DATASET_NAME)
    if args.out_dir is not None:
        out_dir = args.out_dir
    else:
        out_dir = config['out_dir']
    # parameters
    params = config['params']
    if args.seed is not None:
        params['seed'] = int(args.seed)
    if args.epochs is not None:
        params['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        params['batch_size'] = int(args.batch_size)
    if args.init_lr is not None:
        params['init_lr'] = float(args.init_lr)
    if args.lr_reduce_factor is not None:
        params['lr_reduce_factor'] = float(args.lr_reduce_factor)
    if args.lr_schedule_patience is not None:
        params['lr_schedule_patience'] = int(args.lr_schedule_patience)
    if args.min_lr is not None:
        params['min_lr'] = float(args.min_lr)
    if args.weight_decay is not None:
        params['weight_decay'] = float(args.weight_decay)
    if args.print_epoch_interval is not None:
        params['print_epoch_interval'] = int(args.print_epoch_interval)
    if args.max_time is not None:
        params['max_time'] = float(args.max_time)
    
        
    # model parameters
    net_params = config['net_params']
    net_params['device'] = device
    net_params['gpu_id'] = config['gpu']['id']
    net_params['batch_size'] = params['batch_size']
    
    
    if args.full_graph is not None:
        net_params['full_graph'] = True if args.full_graph=='True' else False
    if args.gamma is not None:
        net_params['gamma'] = float(args.gamma)
    if args.m is not None:
        net_params['m'] = int(args.m)
        
      
    if args.LPE is not None:
        net_params['LPE'] = args.LPE
        
        
    if net_params['LPE'] not in ['node', 'edge', 'none']:
        print('[!] User did not provide a valid input argument for \'LPE\'. Valid inputs are \'node\', \'edge\', and \'none\'.')
        exit()
        
    if args.LPE_layers is not None:
        net_params['LPE_layers'] = int(args.LPE_layers)
    if args.LPE_dim is not None:
        net_params['LPE_dim'] = int(args.LPE_dim)
    if args.LPE_n_heads is not None:
        net_params['LPE_n_heads'] = int(args.LPE_n_heads)   
              
    if args.GT_layers is not None:
        net_params['GT_layers'] = int(args.GT_layers)
    if args.GT_hidden_dim is not None:
        net_params['GT_hidden_dim'] = int(args.GT_hidden_dim)
    if args.GT_out_dim is not None:
        net_params['GT_out_dim'] = int(args.GT_out_dim)   
    if args.GT_n_heads is not None:
        net_params['GT_n_heads'] = int(args.GT_n_heads)  
              
    if args.residual is not None:
        net_params['residual'] = True if args.residual=='True' else False
    if args.readout is not None:
        net_params['readout'] = args.readout
    if args.in_feat_dropout is not None:
        net_params['in_feat_dropout'] = float(args.in_feat_dropout)
    if args.dropout is not None:
        net_params['dropout'] = float(args.dropout)
    if args.layer_norm is not None:
        net_params['layer_norm'] = True if args.layer_norm=='True' else False
    if args.batch_norm is not None:
        net_params['batch_norm'] = True if args.batch_norm=='True' else False

    # ZINC
    net_params['num_atom_type'] = dataset.num_atom_type
    net_params['num_bond_type'] = dataset.num_bond_type
    
    #If using full graph, need to add a possible edge type (fake edge)
    if net_params['full_graph']:
        net_params['num_bond_type']+=1
        

    root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_file_name = out_dir + 'results/result_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file

    if not os.path.exists(out_dir + 'results'):
        os.makedirs(out_dir + 'results')
        
    if not os.path.exists(out_dir + 'configs'):
        os.makedirs(out_dir + 'configs')

    train_val_pipeline(MODEL_NAME, dataset, params, net_params, dirs)

    
    
    
main()    
















