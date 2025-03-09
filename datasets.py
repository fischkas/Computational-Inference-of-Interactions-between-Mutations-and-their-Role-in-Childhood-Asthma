#!/usr/bin/env python
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler



class CSVLoader(Dataset):

    def __init__(self, data_path, seperator, train_split, test_split, scale=False, shuffle=True, shuffle_phenotype=False):
        
        dataset = pd.read_csv(data_path, sep = seperator)        
        y = dataset["Phenotype"].values - 1
        x = dataset.drop(columns=["Phenotype"]).values
        x = simulate_missing(x, y)
        

        if shuffle:
            N = x.shape[0]
            idx = np.random.randint(0, N-1, size=N)
            x = x[idx,:]
            y = y[idx]
        
        
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).type(torch.LongTensor)
        

        x_valtest, x_train, y_valtest, y_train = train_test_split(x, y, test_size = train_split, 
                                                                  stratify=y, shuffle=True)
        x_test, x_val, y_test, y_val = train_test_split(x_valtest, y_valtest, test_size=test_split, 
                                                          stratify=y_valtest, shuffle=True)

        self.Xd = {}; self.Yd = {}
        self.Xd["train"] = x_train; self.Yd["train"] = y_train
        self.Xd["val"] = x_val; self.Yd["val"] = y_val
        self.Xd["test"] = x_test; self.Yd["test"] = y_test



    def __getitem__(self, key):

        return self.Xd[key].float(), self.Yd[key].float()

    def get_data(self):

        return self.Xd.float(), self.Yd.float()


    def __len__(self):
        # len(dataset)
        return self.n

    def keys(self):

        return self.Xd.keys()

def simulate_missing(x, y):

    nan_x, nan_y = np.where(np.isnan(x) == True )
    if len(nan_x) > 0:

        cases = np.where(y == 1)[0]
        controls = np.where(y != 1)[0]
        maf_cases = get_MAF(x[cases,:]) # Find mafs for cases
        maf_controls = get_MAF(x[controls,:]) # Find mafs for controls
        case_idx = nan_x <= cases.max() # Find cases
        control_idx = nan_x >= controls.min() # Find controls

        x[ nan_x[case_idx], nan_y[case_idx] ]  = \
            np.random.binomial(n=2, p=maf_cases[nan_y[case_idx]]) # Sample cases

        x[ nan_x[control_idx], nan_y[control_idx] ] = \
            np.random.binomial(n=2, p=maf_controls[nan_y[control_idx]]) # Sample controls
        
        return(x)
    
    else:
        return(x)
    

def get_MAF(genotype_mat):

    n = genotype_mat.shape[0] * 2
    n_copies = np.nansum(genotype_mat, axis=0)
    freqs = np.array([n_copies / n, (n - n_copies) / n])

    return np.max(freqs, axis=0)



def get_class_weight(gwas_data):

    y_train, y_val, y_test = gwas_data["train"][1], gwas_data["val"][1], gwas_data["test"][1]
    N = y_train.shape[0] + y_val.shape[0] + y_test.shape[0]
    ones = y_train.sum() + y_val.sum() + y_test.sum()
    case_r = ones.item()/N
    control_r = 1 - case_r
    return control_r / case_r 
