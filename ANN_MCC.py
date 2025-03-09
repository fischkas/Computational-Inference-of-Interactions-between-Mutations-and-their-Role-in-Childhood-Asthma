#!/usr/bin/env python
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import copy
import warnings
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset


class ANN(nn.Module):

    def __init__(self, num_features,
                 hidden_units,
                 num_classes, 
                 mlp_m=False,
                 main_effect_net_units = [10],
                 act_func = nn.ReLU):
        super(ANN, self).__init__()

        self.neural_network = create_nn([num_features] + hidden_units + [num_classes], act_func)
        self.mlp_m = mlp_m

        if self.mlp_m == True:
            self.univariate_mlps = self.create_main_effect_nets(
                num_features, main_effect_net_units, act_func, False, "uni"
            )

        self.initialize_weights()

    def forward(self, x):

        y = self.neural_network(x)

        if self.mlp_m:
            y += self.forward_main_effect_nets(x, self.univariate_mlps)

        return y

    def create_main_effect_nets(self, num_features, hidden_units, act_func, out_bias, name):
        mlp_list = [
            create_nn([1] + hidden_units + [1], act_func)
            for _ in range(num_features)
        ]
        for i in range(num_features):
            setattr(self, name + "_" + str(i), mlp_list[i])
        return mlp_list

    def forward_main_effect_nets(self, x, mlps):
        forwarded_mlps = []
        for i, mlp in enumerate(mlps):
            forwarded_mlps.append(mlp(x[:, [i]]))
        forwarded_mlp = sum(forwarded_mlps)
        return forwarded_mlp

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)




def create_nn(layer_sizes, act_func):

    ls = list(layer_sizes)
    layers = nn.ModuleList()
    for i in range(1, len(ls) - 1):
        layers.append(nn.Linear(int(ls[i - 1]), int(ls[i])))
        layers.append(act_func())
        #layers.append(nn.BatchNorm1d(int(ls[i])))
        layers.append(nn.Dropout(p=0.01))
    layers.append(nn.Linear(int(ls[-2]), int(ls[-1]), bias=True))

    return nn.Sequential(*layers)

"""
def evaluate(model, data_loader, criterion):
    losses = []
    for inputs, labels in data_loader:
        labels = labels.reshape(-1,1)
        output = model(inputs)
        loss = criterion(output, labels).data
        losses.append(loss)
    return torch.stack(losses).mean()
"""


def MCCLoss_bin(model, data_loader):
    
    # Warnings are ignored due to "matthews_corrcoef()" throwing a misleadning warning
    warnings.filterwarnings("ignore")
    losses = []
    for inputs, labels in data_loader:
        ouputs = model(inputs).sigmoid().round().detach().numpy().flatten()
        loss = matthews_corrcoef(labels.numpy(), ouputs)
        losses.append(loss)
        
    warnings.filterwarnings("default")
    return torch.FloatTensor(losses).mean()

def MCCLoss_mult(model, data_loader):
    losses = []
    for inputs, labels in data_loader:
        labels = labels.detach().numpy()
        output = model(inputs.float()).detach().softmax(dim=1).max(axis=1)[1].float().numpy()
        
        if np.all(output == output[0]):
            loss = 0
        else:
            loss = matthews_corrcoef(labels, output)
        losses.append(loss)

    return torch.FloatTensor(losses).mean()



def PearLoss(model, data_loader):

    losses = []
    for inputs, labels in data_loader:
        labels = labels.reshape(-1,1).detach().numpy().flatten()
        output = model(inputs).detach().numpy().flatten()
        loss = np.corrcoef(labels, output)[0, 1]
        losses.append(loss)
    return torch.FloatTensor(losses).mean()



def create_dataloader(data, batch_size):

    data_loaders = {}
    for k in data.keys():
        x, y = data[k]
        #dataset = TensorDataset(torch.tensor(x), torch.tensor(y)) 
        dataset = TensorDataset(x.detach().clone(), y.detach().clone()) 
        data_loaders[k] = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

    return data_loaders

def test_mcc_bin(data, model):

    x, y_true = data
    y_pred = model(x.float()).sigmoid().round()
    y_pred = y_pred.detach().numpy()
    y_true = y_true.detach().numpy()

    return matthews_corrcoef(y_true, y_pred.flatten())


def test_mcc_mult(data, model):

    x, y_true = data
    
    #x = torch.tensor(x.values)
    #y_true = torch.tensor(y_true.values)

    y_true = y_true.detach().numpy()
    y_pred = model(x).detach().softmax(dim=1).max(axis=1)[1].float().numpy()

    return matthews_corrcoef(y_true, y_pred.flatten())


def get_cor(data, model):

    x, y_true = data
    y_pred = model(x).sigmoid().round()
    y_pred = y_pred.detach().numpy()
    y_true = y_true.detach().numpy()

    return np.corrcoef(y_true, y_pred.flatten())[0, 1]



def train(net,
          dataset,
          batch_size,
          nepochs,
          criterion,
          evaluate,
          test,
          learning_rate,
          l1_const,
          l2_const,
          early_stopping,
          verbose):


    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2_const)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)
    max_divergence = 10
    best_loss = -1
    best_net = None
    divergence = 0
    trainloss_list = []
    valloss_list = []
    data_loaders = create_dataloader(dataset, batch_size)
    model_list = [["x", -1]]
    for epoch in range(nepochs):
        for i, data in enumerate(data_loaders["train"]):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.float()).squeeze()
            train_loss = criterion(outputs, labels.float())
            reg_loss = 0
            for name, param in net.named_parameters():
                if "neural_network" in name and "weight" in name:
                    reg_loss += torch.sum(torch.abs(param))

            (train_loss + reg_loss * l1_const).backward()
            optimizer.step()
            
        trainloss_list.append( train_loss.item() )
        val_loss = evaluate(net, data_loaders["val"])
        scheduler.step(val_loss)
        valloss_list.append(val_loss.item())
        if verbose == True:
            print("Epoch:", epoch+1, "Training loss:", train_loss.item(), "Correlation:", val_loss.item())

        if early_stopping:
            if val_loss >= best_loss:
                best_loss = val_loss
                best_net = copy.deepcopy(net)
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

        
    return trainloss_list, valloss_list, net
