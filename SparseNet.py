import numpy as np
import pandas as pd
import torch
import warnings
import torch.nn as nn
import sparselinear as sl
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef



class SparseNet(nn.Module):

    def __init__(self, 
                 topology_path="connectivty.tsv",
                 hidden_features=50):
        super(SparseNet, self).__init__()

        connections = pd.read_csv(topology_path,
                                  sep='\t', dtype = int)
        snp_features = connections["snp"].max() + 1
        gene_features = connections["gene"].max() + 1
        
        connections = np.array([connections["gene"].values,
                                connections["snp"].values])
        
        connections = torch.tensor(connections).long()
        self.gene_layer = sl.SparseLinear(in_features = snp_features,
                                          out_features = gene_features,
                                          connectivity = connections)# Gene representations, reducing the 
                                                                     # input space using prior knowledge of 
                                                                     # SNP-gene affiliation
        
        self.interaction_layer = nn.Linear(gene_features, hidden_features) # Dense layer to capture non-linear interactions
        self.phenotype_layer = nn.Linear(hidden_features, 1) # Output layer
        
    def forward(self, x):
        
        #self.gene_layer.weights = torch.nn.Parameter(self.gene_layer.weights.abs())
        x = self.gene_layer(x)
        x = self.interaction_layer(x)
        x = self.phenotype_layer(x)
        
        return x.flatten()

  


def MCCLoss_bin(model, data_loader):
    
    # Warnings are ignored due to "matthews_corrcoef()" throwing a misleadning warning
    warnings.filterwarnings("ignore")
    losses = []
    for inputs, labels in data_loader:
        nan_x, nan_y = np.where(np.isnan(inputs) == True )
        inputs[nan_x, nan_y] = inputs[:,nan_y].nanmedian(axis=0)[0]
        
        ouputs = model(inputs).sigmoid().round().detach().numpy().flatten()
        loss = matthews_corrcoef(labels.numpy(), ouputs)
        losses.append(loss)
        
    warnings.filterwarnings("default")
    return torch.FloatTensor(losses).mean()




def test_mcc_bin(data, model):

    x, y_true = data
    y_pred = model(x.float()).sigmoid().round()
    y_pred = y_pred.detach().numpy()
    y_true = y_true.detach().numpy()

    return matthews_corrcoef(y_true, y_pred.flatten())
