# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 08:50:32 2023

@author: 刘明
"""
import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.metrics import roc_curve, matthews_corrcoef
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import shap
import matplotlib.font_manager as font_manager
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import RocCurveDisplay
import joblib


# Machine learning algorithms
def svmhe_grid(X_train, y_train, cv):
    my_svm = svm.SVC(random_state=0, probability=True)
    c_number = []
    for i in range(-4, 4, 2):
        c_number.append(2 ** i)
    gamma = []
    for i in range(-4, 4, 2):
        gamma.append(2 ** i)
    parameters = {'C': c_number, 'gamma': gamma}
    grid_search = GridSearchCV(my_svm, parameters, cv=cv, scoring="f1", return_train_score=False, n_jobs=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def MySVM(X_train, y_train, cv):
    model = svmhe_grid(X_train, y_train, cv)
    auc5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    acc5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    recall5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall').mean()
    precision5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision').mean()
    f15 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1').mean()
    list_cv = [auc5, acc5, recall5, precision5, f15]
    return model, list_cv


def knn_grid(X_train, y_train, cv):
    param_grid = [{ 
        'weights': ['uniform'], 
        'n_neighbors': [i for i in range(1, 11, 2)]},
        {'weights': ['distance'],
        'n_neighbors': [i for i in range(1, 11, 2)],
        'p': [i for i in range(1, 6, 2)]}]
    knn_clf = KNeighborsClassifier()
    grid_search = GridSearchCV(knn_clf,param_grid,cv=cv,scoring="roc_auc", n_jobs=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_



def MyKNN(X_train, y_train,cv):
    model = knn_grid(X_train, y_train, cv)
    auc5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    acc5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    recall5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall').mean()
    precision5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision').mean()
    f15 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1').mean()
    list_cv = [auc5, acc5, recall5, precision5, f15]
    return model, list_cv


def lr_grid(X_train,y_train,cv):
    penaltys = ['l2']#,'l1'
    Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    param_grid = dict(penalty = penaltys, C = Cs)
    clf= LogisticRegression(max_iter=50000)
    grid_search= GridSearchCV(clf,param_grid,cv=cv,scoring='f1',n_jobs=1)
    grid_search.fit(X_train,y_train)
    return grid_search.best_estimator_


def MyLR(X_train, y_train, cv):
    model = lr_grid(X_train,y_train, cv)
    auc5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    acc5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    recall5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall').mean()
    precision5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision').mean()
    f15 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1').mean()
    list_cv = [auc5, acc5, recall5, precision5, f15]
    return model, list_cv



def rf_grid(X_train,y_train,cv):
    param_grid={
    'n_estimators':[100],
    'min_samples_split':[2,4],
    'min_samples_leaf':[1,2,4]
    }
    clf=RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(clf,param_grid,scoring='f1',cv=cv, n_jobs=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def MyRF(X_train, y_train, cv):
    model = rf_grid(X_train,y_train, cv)
    auc5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    acc5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    recall5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall').mean()
    precision5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision').mean()
    f15 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1').mean()
    list_cv = [auc5, acc5, recall5, precision5, f15]
    return model, list_cv


def xgb_grid(X_train, y_train,cv):
    clf = XGBClassifier(eval_metric="logloss",
                        objective="binary:logistic",use_label_encoder=False,
                        seed=42)
    param_grid =  {'max_depth':[8,9,10],
                   'min_child_weight':[1,2,3]}
    grid_search = GridSearchCV(clf,param_grid,scoring='f1',cv=cv, n_jobs=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
            
def MyXGB(X_train, y_train, cv):
    model = xgb_grid(X_train,y_train, cv)
    auc5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    acc5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    recall5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall').mean()
    precision5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision').mean()
    f15 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1').mean()
    list_cv = [auc5, acc5, recall5, precision5, f15]
    return model, list_cv


def gbdt_grid(X_train, y_train,cv):
    clf = GradientBoostingClassifier(random_state=42)
    param_grid =  {'max_depth':[8,9,10],
                   'min_samples_leaf':[50,60,70]}
    grid_search = GridSearchCV(clf,param_grid,scoring='f1',cv=cv, n_jobs=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
            
def MyGBDT(X_train, y_train, cv):
    model = gbdt_grid(X_train,y_train, cv)
    auc5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    acc5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    recall5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall').mean()
    precision5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision').mean()
    f15 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1').mean()
    list_cv = [auc5, acc5, recall5, precision5, f15]
    return model, list_cv


def lgbm_grid(X_train, y_train,cv):
    clf = LGBMClassifier(random_state=42)
    param_grid =  {'learning_rate': [0.01, 0.1, 1],
                   'num_leaves': [29, 30, 31]}
    grid_search = GridSearchCV(clf,param_grid,scoring='f1',cv=cv, n_jobs=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_
            
def MyLGBM(X_train, y_train, cv):
    model = lgbm_grid(X_train,y_train, cv)
    auc5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc').mean()
    acc5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean()
    recall5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='recall').mean()
    precision5 = cross_val_score(model, X_train, y_train, cv=cv, scoring='precision').mean()
    f15 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1').mean()
    list_cv = [auc5, acc5, recall5, precision5, f15]
    return model, list_cv

# Drawing
def multiroc(PATH):
    dict_roc = np.load(os.path.join(PATH,'multimodel_ROC.npy'),allow_pickle=True).item()
    list_color = ['#FB514B','#FB9653','#E6D347','#60E66E','#60DAE6',
                  '#6960E6','#D860E6']
    plt.figure(figsize=(5, 5), dpi=450) 
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False
    for i in range(len(list_multmodel)):
        fpr, tpr, zhi = dict_roc[list_multmodel[i]]
        plt.plot(fpr, tpr, lw=2, label='{} (AUC = {:.3f})'.format(list_multmodel[i], zhi),color = list_color[i])
        plt.plot([0, 1], [0, 1], '--', lw=2, color = 'grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xticks( fontproperties='Times New Roman', size=20) 
        plt.yticks(fontproperties='Times New Roman', size=20) 
        plt.xlabel('False Positive Rate',fontproperties='Times New Roman',fontsize=20)
        plt.ylabel('True Positive Rate',fontproperties='Times New Roman',fontsize=20)
        plt.title('ROC Curve',fontproperties='Times New Roman',fontsize=24)
        font = font_manager.FontProperties(family='Times New Roman', weight='normal', style='normal', size=11)
        plt.legend(loc='lower right', prop=font)
        plt.tight_layout()
    plt.savefig(os.path.join(PATH,'multimodel_ROC.png'), dpi=1200)
    #plt.savefig(os.path.join(PATH,'multimodel_ROC.pdf'))
    plt.close()
    


def roc5(model, X, y, y_train_predprob, j, path5, PATH):
    cv = StratifiedKFold(n_splits=5)
    classifier = model
    X = np.array(X)
    y = np.array(y)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    fig, ax = plt.subplots(figsize=(5, 5))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X[test],
            y[test],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=2,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks( fontproperties='Times New Roman', size=20) 
    plt.yticks(fontproperties='Times New Roman', size=20) 
    plt.xlabel('False Positive Rate',fontproperties='Times New Roman',fontsize=20)
    plt.ylabel('True Positive Rate',fontproperties='Times New Roman',fontsize=20)
    plt.title('ROC Curve',fontproperties='Times New Roman',fontsize=24)
    font = font_manager.FontProperties(family='Times New Roman', weight='normal', style='normal', size=9)
    plt.legend(loc='lower right', prop=font)
    plt.tight_layout()
    plt.savefig(os.path.join(path5, 'train5_AUC.png'), dpi=1200)
    plt.close()

def roc (model, X_test, y_test, y_test_predprob, j, path5, PATH, type1):
    fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)
    roc_auc = auc(fpr, tpr) 
    lw = 2
    plt.figure(figsize=(5, 5),dpi=450)
    plt.rcParams['font.sans-serif'] = ['SimHei'] 
    plt.rcParams['axes.unicode_minus'] = False 
    plt.grid(b = False)
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUC = %0.3f' % roc_auc)  
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks( fontproperties='Times New Roman', size=20) 
    plt.yticks(fontproperties='Times New Roman', size=20) 
    plt.xlabel('False Positive Rate',fontproperties='Times New Roman',fontsize=20)
    plt.ylabel('True Positive Rate',fontproperties='Times New Roman',fontsize=20)
    plt.title('ROC Curve',fontproperties='Times New Roman',fontsize=24)
    plt.legend(loc="lower right",fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(path5,type1+'AUC.png'), bbox_inches='tight', dpi=1200)
    plt.close()   
    dict_auc[list_multmodel[j]] = fpr, tpr, roc_auc
    np.save(os.path.join(PATH,'multimodel_ROC.npy'), dict_auc)
    
    #ConfusionMatrix
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.savefig(os.path.join(path5,type1+'set_matrix.png'), bbox_inches='tight', dpi=1200)
    plt.close()


def best_confusion_matrix(X_test, y_test, y_test_predprob):
    cutoff = 0.5
    y_pred = list(map(lambda x:1 if x>=cutoff else 0,y_test_predprob))
    #TN,FP,FN,TP = confusion_matrix(y_test,y_pred).ravel()
    dict_index, TN,FP,FN,TP = Performance(X_test, y_test, y_pred)
    return y_pred,cutoff,dict_index,TN,FN,FP,TP

def Performance(X_test, y_test, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    tp_index = []
    tn_index = []
    fp_index = []
    fn_index = []
    y_t = list(y_test)
    for i in range(len(y_t)):
        t_label = y_t[i]
        p_label = y_pred[i]
        if t_label == p_label == 0:
            tn += 1
            tn_index.append(i)
        if t_label == p_label == 1:
            tp += 1
            tp_index.append(i)
        if t_label == 0 and p_label == 1:
            fp += 1
            fp_index.append(i)
        if t_label == 1 and p_label == 0:
            fn += 1
            fn_index.append(i)
    if tn*fp*fn*tp == 0:
        tn, fp, fn, tp = 1,1,1,1
    dict_index['tn'] = tn_index
    dict_index['fp'] = fp_index
    dict_index['fn'] = fn_index
    dict_index['tp'] = tp_index
    return dict_index, tn, fp, fn, tp
    

def model_acc(X_test, y_test, y_test_predprob, digits):
    auc = roc_auc_score(y_test, y_test_predprob)
    ap = average_precision_score(y_test, y_test_predprob)
    y_test_pred,test_cutoff,dict_index,TN,FN,FP,TP = best_confusion_matrix(X_test, y_test,y_test_predprob)
    #acc
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    #Sen Spe
    sensetivity = TP/(TP+FN)#true positive rate ,TPR,sensetivity
    specificity = TN/(FP+TN)
    FPR= FP/(FP+TN)
    precision = TP/(TP+FP)
    #PPV NPV
    npv = TN/(FN+TN)
    ppv = TP/(TP+FP)
    #PLR NLR
    plr = (TP/(TP+FN))/(FP/(FP+TN))
    nlr = (FN/(TP+FN))/(TN/(FP+TN))
    #F1
    f1 = f1_score(y_test, y_test_pred)
    #Youden Index
    youden = TP/(TP+FN)+TN/(FP+TN)-1
    #MCC
    mcc = matthews_corrcoef(y_test, y_test_pred)
    #Kappa
    kappa =cohen_kappa_score(y_test_pred, y_test)
    list_tf = [auc, ap, accuracy, sensetivity, specificity, precision, FPR, npv,
               ppv, plr, nlr, f1, youden, mcc, kappa,TN, FN, FP, TP]
    return list_tf
   
def shap_test(model, trainX, testX, number, path5):
    long = len(list(testX.columns))
    if long > 1:
        explainer = shap.Explainer(model, shap.sample(trainX,100))
        shap_values = explainer.shap_values(shap.sample(testX,100),check_additivity=False)
        list_feature = trainX.columns
        shap.summary_plot(shap_values, show = False, max_display=10, plot_type='bar',
                          feature_names=list_feature)
        plt.rcParams['font.sans-serif'] = ['SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  
        plt.xticks( fontproperties='Times New Roman', size=20)
        plt.yticks(fontproperties='Times New Roman', size=20) 
        plt.xlabel('SHAP value (impact on model output)', fontproperties='Times New Roman',fontsize=20)
        plt.tight_layout() 
        plt.savefig(os.path.join(path5,'shap1.png'), dpi=1200)
        plt.close()
        
        explainer = shap.Explainer(model, shap.sample(trainX,100))
        shap_values = explainer(shap.sample(testX,100),check_additivity=False)
        shap.plots.beeswarm(shap_values, show = False, max_display=10)
        plt.rcParams['font.sans-serif'] = ['SimHei']  
        plt.rcParams['axes.unicode_minus'] = False  
        plt.xticks( fontproperties='Times New Roman', size=20) 
        plt.yticks(fontproperties='Times New Roman', size=20) 
        plt.xlabel('SHAP value (impact on model output)', fontproperties='Times New Roman',fontsize=20)
        plt.tight_layout() 
        plt.savefig(os.path.join(path5,'shap2.png'), dpi=1200)
        plt.close()
    

def jianmo_model(trainX, trainY, test_data, test_label, model, j, PATH, path5):
    y_train_predprob = model.predict_proba(trainX)[:,1]   
    y_test_predprob = model.predict_proba(test_data)[:,1]
    y_test_pred = model.predict(test_data)
    list_tf = model_acc(y_test_pred, test_label, y_test_predprob, 3) 
    roc(model, test_data, test_label, y_test_predprob, j, path5, PATH, 'test')
    roc5(model, trainX, trainY, y_train_predprob, j, path5, PATH)
    joblib.dump(model, os.path.join(path5, 'model.pkl'))
    return list_tf
    
def baocun(trainX, trainY, testX, testY, model_name, list_feature, j, df_zhibiao, df_cv, path4, path5, PATH):
    model, list_cv = model_name(trainX, trainY, 5)
    print(list_model_str[j])
    if list_model_str[j] == 'MyLightGBM':
        shap_test(model, trainX, testX, len(list_feature), path5)
    else:
        pass
    list_tf = jianmo_model(trainX, trainY, testX, testY, model, j, PATH, path5)
    list_tf.append(len(list_feature))
    df_zhibiao[list_model_str[j]] = list_tf
    df_cv[list_model_str[j]] = list_cv
    return df_zhibiao, df_cv

# Independent test
def evaluation_select(list_feature, name_neg, j, path4, path5, PATH):
    df_zhibiao = pd.DataFrame()
    df_zhibiao['evaluation'] = list_evaluation
    data_model = pd.read_csv(os.path.join(path4,'trainset_ss.csv'))
    X, y = data_model[list_feature], data_model['label']
    model = joblib.load(os.path.join(path5,"model.pkl"))
    clf = model.fit(X,  y)
    ###########################################
    data = pd.read_csv(os.path.join(path4,'{}_ss.csv'.format(name_neg)))
    X_test, y_test = data[list_feature], data['label']
    y_test_predprob = clf.predict_proba(X_test)[:,1]
    list_duli = model_acc(X_test, y_test, y_test_predprob, 5)
    roc(clf, X_test, y_test, y_test_predprob, j, path5, PATH, 'independent')
    np.save(os.path.join(PATH,'index_independent.npy'), dict_index)
    list_duli.append(len(list_feature))
    df_zhibiao[list_model_str[j]] = list_duli
    df_zhibiao.to_csv(os.path.join(path5,'independentset_evaluation.csv') ,index=None)


def jianmo(list_feature, path4, PATH):
    df_zhibiao = pd.DataFrame()
    df_zhibiao['evaluation'] = list_evaluation
    df_cv = pd.DataFrame()
    df_cv['evaluation'] = ['auc', 'acc', 'sensetivity', 'precision', 'f1']
    data = pd.read_csv(os.path.join(path4,'trainset_ss.csv'))
    trainX = data[list_feature]
    trainY = data['label']
    data1 = pd.read_csv(os.path.join(path4,'testset_ss.csv'))
    testX = data1[list_feature]
    testY = data1['label']
    for j in range(len(list_model_str)):
        path5 = os.path.join(PATH, r'.\{}'.format(list_model_str[j]))
        model_name = list_model[j]                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        if os.path.exists(path5) == False:
            os.mkdir(path5)
        else:
            pass    
        df_zhibiao, df_cv = baocun(trainX, trainY, testX, testY, model_name, list_feature, j, df_zhibiao, df_cv, path4, path5, PATH) 
        evaluation_select(list_feature, 'independent testset', j, path4, path5, PATH)
    df_zhibiao.to_csv(os.path.join(PATH,'testset_evaluation.csv') ,index=None)
    df_cv.to_csv(os.path.join(PATH,'train_evaluation.csv') ,index=None)
    multiroc(PATH)
    
        

dict_auc = {}
dict_prc = {}
dict_index = {}
list_evaluation = ['auc', 'ap', 'accuracy', 'sensetivity', 'specificity', 'precision', 
                   'FPR','npv','ppv', 'plr', 'nlr', 'f1', 'youden', 
                   'mcc', 'kappa','TN', 'FN', 'FP', 'TP', 'number']

list_model = [MyLR, MySVM, MyKNN, MyRF, MyXGB, MyGBDT, MyLGBM]
list_model_str = ['MyLR', 'MySVM', 'MyKNN', 'MyRF', 'MyXGBoost', 'MyGBDT', 'MyLightGBM']
list_multmodel = ['LR', 'SVM', 'KNN', 'RF', 'XGBoost', 'GBDT', 'LightGBM']

path1 = os.getcwd()
path2 = r'./normalization'

list_feature = []

path3 = r'./result'
if os.path.exists(path3) == False:
    os.mkdir(path3)
else:
    pass
if len(list_feature) == 0:
    df = pd.read_csv(os.path.join(path1,'data.csv'))
    column_names = df.columns.tolist()
    column_names.remove('label')
    list_feature = list(column_names)
    jianmo(list_feature, path2, path3)
else:
    jianmo(list_feature, path2, path3)

