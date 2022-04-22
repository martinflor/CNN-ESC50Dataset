# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 18:50:10 2022

@author: Florian Martin

Dataset - computation - shaping

"""


import pandas as pd

class DS() : 
    def __init__(self) :
        labels = ['datas/crackling_fire.txt', 'datas/chirping_birds.txt', 'datas/helicopter.txt', 'datas/chainsaw.txt', 'datas/handsaw.txt']
        frames = []
        for label_ in labels :
            frames.append(self.add_label(self.txt_to_dataframe(label_)[:5000], label_))
            
    
        self.df = pd.concat(frames)
        
        
    def CSVFormat(self, filename) :
        self.df.to_csv('datas/' + filename, index=False)

    def txt_to_dataframe(self, filename) :
        
        f = open(filename, 'r')
        
        dic = {}
        i = 0
        X_i = []
        
        while True :
            line = f.readline()
            if not line :
                break
            
            if line == '\n' :
                dic[i] = X_i
                i += 1
                X_i = []
                
            else :
                X_i.append(float(line.replace('\n','')))
               
        
        return pd.DataFrame.from_dict(dic).T

    def label_encoder(self, label) :
    
        if (label == 'datas/crackling_fire.txt') :
            return 0
        elif (label == 'datas/chirping_birds.txt') :
            return 1
        elif (label == 'datas/helicopter.txt') :
            return 2
        elif (label == 'datas/chainsaw.txt') :
            return 3
        else :
            return 4
            
    def add_label(self, dataframe, label) :
        
        dataframe['label'] = self.label_encoder(label)
        
        return dataframe




