#!/usr/bin/env python
import dask
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from pandas_plink import read_plink1_bin


class SmartPlink(Dataset):
    
    def __init__(self, geno_path, pheno_path):
        
        G = read_plink1_bin(geno_path + ".bed",
                            geno_path + ".bim",
                            geno_path + ".fam",
                            verbose=False)
                
        sample_ids = G.sample.iid.values
        pheno_file = pd.read_csv(pheno_path, sep='\t')
        
        # Get case coordinates and assign labels
        cases = pheno_file[pheno_file["Phenotype"] == 2]["IID"].values
        controls = pheno_file[pheno_file["Phenotype"] == 1]["IID"].values
        isin_cases = np.isin(sample_ids, cases)
        idx_cases = np.where(isin_cases)[0]
        G.trait.values[isin_cases] = 1

        # Get control coordinates and assign labels
        isin_controls = np.isin(sample_ids, controls)
        idx_controls = np.where(isin_controls)[0]
        G.trait.values[isin_controls] = 0

        # Gather indices and invert matches to delete rows in xarray (this is more memory efficient than extracting)
        idx_phenotype = np.concatenate((idx_cases, idx_controls), axis=0)
        inverted_idx = np.isin(G.sample.values, G.sample.values[idx_phenotype], invert=True)
        G = G.drop_sel(sample = G.sample.values[inverted_idx])
        
        # Get length of phenotype matched genotypes
        self.n = G.shape[0]

        # Split data in training, validation & testing
        trait = G.trait.values.astype(float)
        idx = np.arange(start=0, stop=self.n, step=1)
        x_train, x_test, y_train, y_test = train_test_split(idx, trait, train_size = 0.8, stratify=trait, shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = 0.8, stratify=y_train, shuffle=True)
        
        self.Xd = {}; self.Yd = {}

        self.Xd["train"] = G[np.sort(x_train), :]
        self.Yd["train"] = y_train
        
        self.Xd["val"] = G[np.sort(x_val), :]
        self.Yd["val"] = y_val
        
        self.Xd["test"] = G[np.sort(x_test), :]
        self.Yd["test"] = y_test


    def __getitem__(self, key):
        
        return self.Xd[key], self.Yd[key]


    def __len__(self):
        # len(dataset)
        return self.n

    def keys(self):

        return self.Xd.keys()


class LoadSmartPlink(Dataset):
    
    def __init__(self, data_split):
        
        self.genotype, self.phenotype = data_split
        self.n = self.phenotype.shape[0]

    def __getitem__(self, item):
        
        return torch.from_numpy(self.genotype[item,:].values), torch.tensor(self.phenotype).float()[item]


    def __len__(self):
        # len(dataset)
        return self.n


        

class PlinkDataset(Dataset):

    def __init__(self, G, train_split, test_split, 
                 scale=False, shuffle=True, shuffle_phenotype=False):

        x = G.values
        y = G.trait.values.astype(int) - 1
        


        if shuffle_phenotype:
            N = x.shape[0]
            ydx = np.random.randint(0, N-1, size=N)
            y = y[ydx]

        
        if scale:
            
            x = x / x.max()
        
        
        
        x = torch.from_numpy(x.astype(int))
        y = torch.from_numpy(y.astype(int)).type(torch.LongTensor)
        
        
        self.Xd = {}; self.Yd = {}

        if train_split == 1.0:
            
            self.Xd["train"] = x; self.Yd["train"] = y

            
        else:
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
    
    
    
class PreSplit(Dataset):

    def __init__(self, G_train, G_val, G_test, 
                 scale=False, shuffle=True, shuffle_phenotype=False):

        
        PlinkSplitsLabels = ["train", "val", "test"] 
        PlinkSplits = [G_train, G_val, G_test]
        self.Xd = {}; self.Yd = {}
        for i in range(0, len(PlinkSplits)):
            x = PlinkSplits[i].values
            y = PlinkSplits[i].trait.values.astype(int)
            
            if scale:
                x = x / x.max()
            
            if shuffle_phenotype:
                N = x.shape[0]
                ydx = np.random.randint(0, N-1, size=N)
                y = y[ydx]
                      
            x = torch.from_numpy(x.astype(int))
            y = torch.from_numpy(y.astype(int)).type(torch.LongTensor)

            self.Xd[PlinkSplitsLabels[i]] = x
            self.Yd[PlinkSplitsLabels[i]] = y



    def __getitem__(self, key):

        return self.Xd[key].float(), self.Yd[key].float()

    def get_data(self):

        return self.Xd.float(), self.Yd.float()


    def __len__(self):
        # len(dataset)
        return self.n

    def keys(self):

        return self.Xd.keys()


def merge_geno_pheno(geno_path, pheno_path, delim="\t"):
    
    
    G = read_plink1_bin(geno_path+".bed", 
                        geno_path+".bim", 
                        geno_path+".fam", 
                      verbose=False)
    
    sample_ids = G.sample.iid.values
    pheno_file = pd.read_csv(pheno_path, sep=delim)

    # Get case coordinates and assign labels
    cases = pheno_file[pheno_file["Phenotype"] == 2]["IID"].values
    controls = pheno_file[pheno_file["Phenotype"] == 1]["IID"].values
    isin_cases = np.isin(sample_ids, cases)
    idx_cases = np.where(isin_cases)[0]
    G.trait.values[isin_cases] = 1

    # Get control coordinates and assign labels
    isin_controls = np.isin(sample_ids, controls)
    idx_controls = np.where(isin_controls)[0]
    G.trait.values[isin_controls] = 0

    # Gather indices and invert matches to delete rows in xarray (this is more memory efficient than extracting)
    idx_phenotype = np.concatenate((idx_cases, idx_controls), axis=0)
    inverted_idx = np.isin(G.sample.values, G.sample.values[idx_phenotype], invert=True)
    G = G.drop_sel(sample = G.sample.values[inverted_idx])
    
    return G
    

def get_class_weight(gwas_data):

    #y_train, y_val, y_test = gwas_data["train"][1], gwas_data["val"][1], gwas_data["test"][1]
    #N = y_train.shape[0] + y_val.shape[0] + y_test.shape[0]
    y_train, y_val = gwas_data["train"][1], gwas_data["val"][1]
    N = y_train.shape[0] + y_val.shape[0] 
    ones = y_train.sum() + y_val.sum()
    case_r = ones.item()/N
    control_r = 1 - case_r
    return control_r/ case_r 


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
