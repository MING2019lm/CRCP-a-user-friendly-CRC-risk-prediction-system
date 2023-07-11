# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:51:42 2022

@author: 刘明
"""
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LassoCV   
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import SelectKBest,chi2,SelectFromModel,f_classif,mutual_info_classif
from itertools import chain
from sklearn.svm import LinearSVC

#Enter the name of the file for normalization
def guiyihua1 (name_biaozhun, list_feature, path3, path4):
    alldatas = pd.read_csv(os.path.join(path3,'{}.csv'.format(name_biaozhun)))
    newdatas = alldatas[list_feature]
    feature = newdatas.columns
    mean1 = []
    std1 = []
    for i in feature:
        mean1.append(round(alldatas[i].mean(),2))
        std1.append(round(alldatas[i].std(),2))
    ss = pd.DataFrame()
    ss['mean'] = mean1
    ss['std'] = std1
    ss.index = feature
    ss.to_csv(os.path.join(path4,'{}_normalization.csv'.format(name_biaozhun)))
    return feature, ss


def guiyihua2(name_pos, list_feature, ss, path3, path4):
    alldatas = pd.read_csv(os.path.join(path3,'{}.csv'.format(name_pos)))
    df = pd.DataFrame()
    df['label'] = alldatas['label']
    for i in list_feature:
        df[i] = (alldatas[i] - ss.loc[i,'mean'])/ss.loc[i,'std']
    df.to_csv(os.path.join(path4, "{}_ss.csv".format(name_pos)),index=None)
    
    
def guiyihua(name_pos, list_all_datas, list_feature, path3, path4):
    feature, ss = guiyihua1(name_pos, list_feature, path3, path4)
    for i in list_all_datas:
        guiyihua2(i, feature, ss, path3, path4)

#Feature selection
def MyLASSO(X_train, y_train, cv,list_feature,number):
    Lambdas=[2,0.1,1,0.001,0.0005]
    lasso_cv=LassoCV(alphas=Lambdas,cv=cv,random_state=0,max_iter=100000)
    lasso_cv.fit(X_train,y_train)
    importance = abs(lasso_cv.coef_)
    list_paixu = shunxu(importance, list_feature, 'LASSO', number)
    model_select = SelectFromModel(lasso_cv, prefit=True)
    dict_feature['LASSO'] = list(model_select.get_feature_names_out(list_paixu))

def MySVM(X_train, y_train, cv,list_feature,number):
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)
    importances = lsvc.coef_
    importances = abs(np.array(list(chain.from_iterable(importances))))
    list_paixu = shunxu(importances, list_feature, 'SVM', number)
    model_select = SelectFromModel(lsvc, prefit=True)
    dict_feature['SVM'] = list(model_select.get_feature_names_out(list_paixu))

def MyLR(X_train, y_train, cv,list_feature,number):
    grid_search= LogisticRegression(max_iter=50000)
    grid_search.fit(X_train, y_train)
    model = grid_search
    importances = model.coef_
    importances = abs(np.array(list(chain.from_iterable(importances))))
    list_paixu = shunxu(importances, list_feature, 'LR', number)
    model_select = SelectFromModel(model, prefit=True)
    dict_feature['LR'] = list(model_select.get_feature_names_out(list_paixu))

def MyRF(X_train,y_train,cv,list_feature,number):
    param_grid={
    'n_estimators':[100],
    'min_samples_split':[2,4],
    'min_samples_leaf':[1,2,4]
    }
    clf=RandomForestClassifier()
    grid_search = GridSearchCV(clf,param_grid,scoring='roc_auc',cv=cv, n_jobs=1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    importances = abs(model.feature_importances_)
    list_paixu = shunxu(importances, list_feature, 'RF', number)
    model_select = SelectFromModel(model, prefit=True)
    dict_feature['RF'] = list(model_select.get_feature_names_out(list_paixu))
            
def MyXGBT(X_train,y_train,cv,list_feature,number):
    clf = XGBClassifier(eta=0.01,subsample=0.5,eval_metric="logloss",objective="binary:logistic",use_label_encoder=False)
    param_grid =  {'max_depth':[8,9,10],
                   'min_child_weight':[1,2,3]}
    grid_search = GridSearchCV(clf,param_grid,scoring='roc_auc',cv=cv, n_jobs=1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    importances = abs(model.feature_importances_)
    list_paixu = shunxu(importances, list_feature, 'XGBT', number)
    model_select = SelectFromModel(model, prefit=True)
    dict_feature['XGBT'] = list(model_select.get_feature_names_out(list_paixu))

def MyGDBT(X_train,y_train,cv,list_feature,number):
    clf = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1, 
    subsample=0.8,
    loss='deviance',
    max_features='sqrt',
    criterion='friedman_mse',
    min_samples_split =1200, 
    min_impurity_decrease=0.0,
    #max_depth=7,
    max_leaf_nodes=None,
    #min_samples_leaf =60, 
    warm_start=False,
    random_state=10)
    param_grid =  {'max_depth':[8,9,10],
                   'min_samples_leaf':[50,60,70]}
    grid_search = GridSearchCV(clf,param_grid,scoring='roc_auc',cv=cv, n_jobs=1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    importances = abs(model.feature_importances_)
    list_paixu = shunxu(importances, list_feature, 'GDBT', number)
    model_select = SelectFromModel(model, prefit=True)
    dict_feature['GDBT'] = list(model_select.get_feature_names_out(list_paixu))

def MyLGBM(X_train,y_train,cv,list_feature,number):
    clf = LGBMClassifier()
    param_grid =  {'learning_rate': [0.01, 0.1, 1],
                   'num_leaves': [29, 30, 31]}
    grid_search = GridSearchCV(clf,param_grid,scoring='roc_auc',cv=cv, n_jobs=1)
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    importances = abs(model.feature_importances_)
    list_paixu = shunxu(importances, list_feature, 'LGBM', number)
    model_select = SelectFromModel(model, prefit=True)
    dict_feature['LGBM'] = list(model_select.get_feature_names_out(list_paixu))

def shunxu(importances, list_feature, name, number):
    a = list(np.argsort(-importances))
    list_paixu = []
    for i in a :
        list_paixu.append(list_feature[i])
    df_ordervale[name] = importances
    df_feature_order[name] = list_paixu[:number]
    return list_paixu


def MyCLASSIF(X_train,y_train,k_value,list_feature,number):
    c = SelectKBest(f_classif,k=k_value)
    model = c.fit(X_train,y_train)
    importances = -np.log10(model.pvalues_)
    list_paixu = qu_mean(importances, list_feature, 'IF', number)
    dict_feature['IF'] = list_paixu

def MyMI(X_train,y_train,list_feature,number):
    importances = mutual_info_classif(X_train,y_train)
    list_paixu = qu_mean(importances, list_feature, 'MI', number)
    dict_feature['MI'] = list_paixu
    
def qu_mean(importances, list_feature, name, number):
    mean_ = np.mean(importances)
    c=np.sum(importances>=mean_)
    a = list(np.argsort(-importances))
    list_paixu = []
    for i in a :
        list_paixu.append(list_feature[i])
    df_ordervale[name] = importances
    df_feature_order[name] = list_paixu[:number]
    return list_paixu[:c]
    
def MyALL(list_feature):
    dict_feature['ALL'] = list_feature
    
def fselect(path4, list_feature):
    train = pd.read_csv(os.path.join(path4,'trainset_ss.csv'))
    X_train, y_train = train.drop(['label'],axis=1), train['label']
    cv = 10
    number = len(list_feature)
    MyCLASSIF(X_train,y_train,'all',list_feature,number)
    MyMI(X_train,y_train,list_feature,number)
    MyLASSO(X_train,y_train,cv,list_feature,number)
    MyLR(X_train, y_train, cv,list_feature,number)
    MySVM(X_train, y_train, cv,list_feature,number)
    MyRF(X_train,y_train,cv,list_feature,number)
    MyXGBT(X_train,y_train,cv,list_feature,number)
    MyGDBT(X_train,y_train,cv,list_feature,number)
    MyLGBM(X_train,y_train,cv,list_feature,number)
    MyALL(list_feature)
    df_ordervale['feature'] = list_feature
    df_ordervale.to_csv(os.path.join(path4,'feature_value_{}.csv'.format(number)),index=None)
    df_feature_order.to_csv(os.path.join(path4,'feature_order_{}.csv'.format(number)),index=None)
    np.save(os.path.join(path4,'feature_order'), dict_feature)

df_ordervale = pd.DataFrame()
df_feature_order = pd.DataFrame()
dict_feature = {}

path1 = os.getcwd()
path2 = r'./normalization'

if os.path.exists(path2) == False:
    os.mkdir(path2)
else:
    pass

# Extraction of datasets, normalisation, modelling, filtering of features modelling plus evaluation
df = pd.read_csv(os.path.join(path1,'data.csv'))
# Of features beyond the label
column_names = df.columns.tolist()
column_names.remove('label')
list_feature = list(column_names)
# Split training set test set
train, test= train_test_split(df, test_size=0.2, random_state=42)
train.to_csv(os.path.join(path1,'trainset.csv'),index=None)
test.to_csv(os.path.join(path1,'testset.csv'),index=None)
# Normalised data sets
# Please place all datasets to be normalised into the dataset folder and write the names in the list_biaozhun list.
list_biaozhun = ['trainset', 'testset', 'independent testset']
guiyihua('trainset', list_biaozhun, list_feature, path1, path2)
fselect(path2, list_feature)
