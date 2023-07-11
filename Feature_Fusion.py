# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 10:34:38 2023

@author: 刘明
"""

import pandas as pd
import os
path1 = os.getcwd()

def ronghe1(name):
    list_rong = ['RBC','HCT','HGB','RDW-CV',
          'PLT','PCT','PDW',
          'WBC','MON#','NEU#','LYM#','BAS#','EOS#']
    long = len(list_rong)
    list_rong2 = ['label','sex','age',
          'RBC','HCT','HGB','RDW-CV',
          'PLT','PCT','PDW',
          'WBC','MON#','NEU#','LYM#','BAS#','EOS#']
    data = pd.read_csv(os.path.join(path1,name)) 
    df1 = data[list_rong]
    df1['RBC'] = data['RBC']*10
    df1['HCT'] = data['HCT']
    df1['HGB'] = data['HGB']
    df1['RDW-CV'] = data['RDW-CV']
    df1['PLT'] = data['PLT']/10
    df1['PCT'] = data['PCT']*100
    df1['PDW'] = data['PDW']*10
    df1['WBC'] = data['WBC']*10
    df1['MON#'] = data['MON#']*100
    df1['NEU#'] = data['NEU#']*10
    df1['LYM#'] = data['LYM#']*100
    df1['BAS#'] = data['BAS#']*1000
    df1['EOS#'] = data['EOS#']*1000
    df1 = df1[list_rong]
    print(df1.columns)
    df = data[list_rong2]
    for i in range(long):
        for j in range(long-2):
            if i!=j :
                df[str(list_rong[i])+'/'+str(list_rong[j])] = df1.iloc[:, i]/df1.iloc[:, j]
    for i in range(long):
        for j in range(i+1, long):
            df[str(list_rong[i])+'*'+str(list_rong[j])] = df1.iloc[:, i]*df1.iloc[:, j]
    return df

def ronghe2(df, name):
    list_2 = ['MON#/WBC','EOS#/WBC','NEU#/WBC',
                'LYM#/WBC','BAS#/WBC',
          'HCT/RBC','HGB/RBC','HGB/HCT','PCT/PLT']
    list_2name = ['MON%','EOS%','NEU%',
                'LYM%','BAS%',
          'MCV','MCH','MCHC','MPV']
    number = len(list_2)
    for i in range(number):
        df.rename(columns={list_2[i]:list_2name[i]},inplace=True)
    df['MCV'] = df['MCV']*100
    df['MCH'] = df['MCH']*10
    df['MCHC'] = df['MCHC']*100
    df['MPV'] = df['MPV']*10
    df['MON%'] = df['MON%']*10
    df['EOS%'] = df['EOS%']
    df['NEU%'] = df['NEU%']*100
    df['LYM%'] = df['LYM%']*10
    df['BAS%'] = df['BAS%']
    df.to_csv(os.path.join(path1,name+"_fusion.csv"),index=None)
    return df


df1 = ronghe2(ronghe1('data.csv'), 'data')
df2 = ronghe2(ronghe1('independent testset.csv'), 'independent testset')