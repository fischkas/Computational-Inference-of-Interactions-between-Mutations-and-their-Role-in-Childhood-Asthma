#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from SparseNet import *
from plink_datasets import *


# In[81]:


#dask.config.set({"array.slicing.split_large_chunks": True})

geno_path = "/home/user/directory/plink_data/genotypes/"
pheno_path = "/home/user/directory/phenotypes/hos6.pheno.txt"

geno_data = SmartPlink(geno_path, pheno_path)

batch_size = 128
train_split = LoadSmartPlink(geno_data["train"])
train_loader = DataLoader(train_split, batch_size = batch_size)

val_split = LoadSmartPlink(geno_data["val"])
val_loader = DataLoader(val_split, batch_size = batch_size)

test_split = LoadSmartPlink(geno_data["test"])
test_loader = DataLoader(test_split, batch_size = batch_size)


# In[59]:


net = SparseNet(topology_path="/home/user/Phenotype_prediction/connectivty.tsv",
                hidden_features=500)


# In[ ]:


nepochs = 150
l1_const=5e-5
l2_const=1e-2
early_stopping = True
verbose = True
max_divergence = 10
best_loss = -1
learning_rate = 1e-4
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2_const)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(14.3))
evaluate = MCCLoss_bin
test = test_mcc_bin
best_net = None
divergence = 0
trainloss_list = []
valloss_list = []
model_list = [["x", -1]]
for epoch in range(nepochs):
    for i, data in enumerate(train_loader):
        inputs, labels = data
        nan_x, nan_y = np.where(np.isnan(inputs) == True )
        inputs[nan_x, nan_y] = inputs[:,nan_y].nanmedian(axis=0)[0]
        optimizer.zero_grad()
        
        outputs = net(inputs)
        train_loss = criterion(outputs, labels)
        reg_loss = 0
        for name, param in net.named_parameters():
            if "neural_network" in name and "weight" in name:
                reg_loss += torch.sum(torch.abs(param))

        (train_loss + reg_loss * l1_const).backward()
        optimizer.step()

        
        
        
        
    trainloss_list.append( train_loss.item() )
    val_loss = evaluate(net, val_loader)
    scheduler.step(val_loss)
    valloss_list.append(val_loss.item())
    
    if verbose == True:
        print("Epoch:", epoch+1, "Training loss:", train_loss.item(), "Correlation:", val_loss.item())

    if early_stopping:
        if val_loss >= best_loss:
            best_loss = val_loss
            best_net = copy.deepcopy(net)
            
            save_point = "/home/kasper/nas/model_runs/sparse_net/sparse_model_epoch_" + str(epoch+1) + ".pth"
            torch.save(best_net, save_point)
            
            model_list.append([best_net, best_loss.item()])
            divergence = 0
        else:
            divergence += 1
            if divergence > max_divergence:
                print("Early stopping!")
                sorted_models = sorted(model_list, key=lambda x: x[1], reverse=True)[0]
                net = sorted_models[0]
                val_loss = sorted_models[1]
                break


trainloss_file = open("/home/kasper/nas/model_runs/sparse_net/training_progress/training_loss.txt", "w")
for element in trainloss_list:
    trainloss_file.write(element + "\n")
trainloss_file.close()

valloss_file = open("/home/kasper/nas/model_runs/sparse_net/training_progress/validation_loss.txt", "w")
for element in valloss_file:
    valloss_file.write(element + "\n")
valloss_file.close()
                
model_path = "/home/kasper/nas/model_runs/sparse_net/sparse_gene_layer.pth"
torch.save(net, model_path)


# In[102]:


test_n = geno_data["test"][0].shape[0]
test_performance = np.empty([test_n, 2])
i = 0
j = batch_size
for inputs, labels in test_loader:
    
    nan_x, nan_y = np.where(np.isnan(inputs) == True)
    inputs[nan_x, nan_y] = inputs[:,nan_y].nanmedian(axis=0)[0]
    
    outputs = net(inputs).sigmoid().detach()

    batch_peformance = np.concatenate((outputs.reshape(-1,1), labels.numpy().reshape(-1,1)), axis=1)
    test_performance[i:j,:] = batch_peformance

    i += batch_size
    j += batch_size

np.savetxt("/home/kasper/nas/model_runs/sparse_net/performance/test_predictions.csv", test_performance, delimiter=",")


# In[ ]:


train_n = geno_data["train"][0].shape[0]
train_performance = np.empty([train_n, 2])
i = 0
j = batch_size
for inputs, labels in train_loader:
    
    nan_x, nan_y = np.where(np.isnan(inputs) == True)
    inputs[nan_x, nan_y] = inputs[:,nan_y].nanmedian(axis=0)[0]
    
    outputs = net(inputs).sigmoid().detach()

    batch_peformance = np.concatenate((outputs.reshape(-1,1), labels.numpy().reshape(-1,1)), axis=1)
    train_performance[i:j,:] = batch_peformance

    i += batch_size
    j += batch_size


np.savetxt("/home/kasper/nas/model_runs/sparse_net/performance/train_predictions.csv", train_performance, delimiter=",")


