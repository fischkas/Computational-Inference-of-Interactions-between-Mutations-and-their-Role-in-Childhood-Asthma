#!/usr/bin/env python
import torch
import tables
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pandas_plink import read_plink1_bin


class load_hdf5(Dataset):

    def __init__(self, geno_path, pheno_path, split):
        
        self.geno_path = geno_path
        self.pheno_path = pheno_path
        subjects = pd.read_csv(self.pheno_path, sep = ",")
        self.subjects = subjects[subjects["set"] == split]
        
        
    def __getitem__(self, item):
       
        h5file = tables.open_file(self.geno_path, "r")
        
        geno_row = np.array(self.subjects["genotype_row"].iloc[item], dtype=np.float64)
        data = h5file.root.data[geno_row, :]
        h5file.close()

        labels = self.subjects["labels"].iloc[item]
        

        return data, labels
    
    
    def __len__(self):
        return len(self.subjects)
        
        
        
class load_csv(Dataset):

    def __init__(self, geno_path, pheno_path, split):
        
        self.geno_path = geno_path
        self.pheno_path = pheno_path
        subjects = pd.read_csv(self.pheno_path, sep = ",")
        self.genomat = pd.read_csv(self.geno_path, sep = ",")       
        self.subjects = subjects[subjects["set"] == split]
        
    def __getitem__(self, item):
        
        geno_row = self.subjects["genotype_row"].iloc[item]
        data = self.genomat.drop(["FID", "IID"], axis=1).iloc[geno_row,:].values

        labels = self.subjects["labels"].iloc[item]


        return data, labels
    
    
    def __len__(self):
        return len(self.subjects)
        
        
        
