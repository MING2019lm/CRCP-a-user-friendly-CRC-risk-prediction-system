# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 18:49:50 2023

@author: 刘明
"""

import pandas as pd
import os
path1 = os.getcwd()
path3 = r'.\result'
if os.path.exists(path3) == False:
    os.mkdir(path3)
else:
    pass
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# p_value
df = pd.read_csv(os.path.join(path1,'trainset.csv')) 
positive_subset = df[df['label'] == 1]
negative_subset = df[df['label'] == 0]
t_test_results = pd.DataFrame(columns=['feature', 't_value', 'p_value'])
for feature in df.columns:
    if feature == 'label':
        continue
    t, p = stats.ttest_ind(positive_subset[feature], negative_subset[feature], equal_var=False)
    t_test_results = t_test_results.append({'feature': feature, 't_value': t, 'p_value': p}, ignore_index=True)

t_test_results['p_value'] = t_test_results['p_value'].apply(lambda x: '<0.01' if x < 0.01 else '{:.3f}'.format(x))
t_test_results.to_csv(os.path.join(path3,'p_value.csv'),index=None)

# MinMaxScaler
def trim_outliers(data):
    q_5 = data.quantile(0.05)
    q_95 = data.quantile(0.95)
    
    for col in data.columns:
        data[col] = np.where(data[col] < q_5[col], q_5[col], data[col])
        data[col] = np.where(data[col] > q_95[col], q_95[col], data[col])
    
    return data

def MM_guiyihua ():
    df = pd.read_csv(os.path.join(path1,'trainset.csv')) 
    column_names = df.columns.tolist()
    column_names.remove('label')
    list_feature = list(column_names)
    df1 = df[list_feature]
    df1 = trim_outliers(df1)
    model = MinMaxScaler(feature_range=(-1, 1))
    df1 = model.fit_transform(df1)
    df2 = pd.DataFrame(df1) 
    df2.columns = list_feature
    df2['label'] = df['label']
    df2.to_csv(os.path.join(path3,"mm.csv"),index=None)
    return df2
   
df = MM_guiyihua()


# piersen
def piersen():
    data = pd.read_csv(os.path.join(path1,'trainset.csv')) 
    del data['label']
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    mask = np.tril(np.ones_like(data.corr(), dtype=bool))
    sns_plot = sns.heatmap(data.corr(), cmap=sns.diverging_palette(20, 220, n=200), 
                           mask=mask, annot=False, center=0, ax=ax, annot_kws={"size": 15})
    cbar = sns_plot.collections[0].colorbar
    cbar.ax.tick_params(labelsize=22)
    cbar.ax.set_ylabel('', fontsize=22) 
    plt.xticks(fontproperties='Times New Roman', size=22, rotation=45)
    plt.yticks(fontproperties='Times New Roman', size=22, rotation=0)
    plt.close()
    fig.savefig(os.path.join(path3,'piersen_picture.png'), dpi=1200, bbox_inches='tight')

piersen()