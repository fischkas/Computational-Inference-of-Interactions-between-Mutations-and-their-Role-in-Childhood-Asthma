#!/usr/bin/env python



import numpy as np
import pandas as pd
import torch
from NID_fast import *
import pickle



for i in range(1, 101):
    model_path = "/path/to/models/model_" + str(i) + ".pth"
    model = torch.load(model_path)
    model_weights = get_weights(model)
    
    ### Pairwise interactions
    w_input, w_later = preprocess_weights(model_weights)
    pairwise = get_nlargest(generator = interpret_pairwise_interactions(w_input, w_later), N = 100)
    pairwise_pd = pd.DataFrame(pairwise, columns=["Index", "Importance"])
    
    pair_path = "/path/to/pairs/model_" + str(i) + ".csv"
    pairwise_pd.to_csv(pair_path)
    
    ### Higher order interactions
    higher = get_interactions(model_weights)
    higher_pd = pd.DataFrame(higher, columns=["Index", "Importance"])
    
    higher_path = "/path/to/higher/model_" + str(i) + ".csv"
    higher_pd.to_csv(higher_path)



