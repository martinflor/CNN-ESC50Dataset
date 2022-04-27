# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 19:05:05 2022

@author: Florian


Pytorch format Dataset

"""

import pandas as pd
import torch
from torch.utils.data import Dataset


class MSDS(Dataset):
  def __init__(self, path):
    self.df = pd.read_csv(path)
            
  def __len__(self):
    return len(self.df)    
    
  def __getitem__(self, idx):
    
    specgram = torch.Tensor(self.df.iloc[idx, 0:200].values.reshape(20,10)).unsqueeze(0)
    class_label = torch.tensor(self.df.iloc[idx, 200])
    

    return specgram, class_label

