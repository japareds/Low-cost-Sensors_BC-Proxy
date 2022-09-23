#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBJECTIVE: 
    BFS and model hyperparamters optimization
    
Created on Mon Jul 11 10:53:21 2022

@author: jparedes
"""
import numpy as np
import pandas as pd
import time
import os

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split, KFold,cross_val_score, cross_validate,learning_curve, validation_curve,GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS 
import joblib
import csv

import BCP_DataSet as DS
import BCP_PreProcessing as PP
#%%
def training_split(df,test_frac):
    X_tot = df.iloc[:,1:]
    Y_tot = df.iloc[:,0]
    test_frac = 0.25    
    X_train, X_test, Y_train, Y_test = train_test_split(X_tot,Y_tot,test_size=test_frac,
                                                        random_state=92,shuffle=True)
    return X_train,X_test,Y_train,Y_test
#%%
def best_estimator_predict(gs_results,X_train,X_test,Y_train,Y_test):
    """
    Predict training and testing set performance
    using the best model found via grid search

    Parameters
    ----------
    gs_results : gridSearchCV obj
                gridsearch object
                
    X_train : pandas DataFrame
            predictors training set
            
    X_test : pandas DataFrame
            predictors testing set
            
    Y_train : Pandas Series
            BC training set
            
    Y_test : Pandas series
            BC testing set

    Returns
    -------
    RMSE_train : float
            RMSE training set
    R2_train : float
            R2 training set
    adj_R2_train : float
            adjusted R2 training set
            
    RMSE_test : float
            RMSE testing set
    R2_test : float
            R2 testing set
    adj_R2_test : float
            adjusted R2 testing set

    """
    # use GS best model or define it yourself
    use_gs_results = True
    if use_gs_results:
        best_model = gs_results.best_estimator_
    # training set scores
    y_pred = best_model.predict(X_train)
    RMSE_train = np.sqrt(mean_squared_error(Y_train,y_pred))
    R2_train = r2_score(Y_train,y_pred)
    adj_R2_train = 1-(1-R2_train)*(X_train.shape[0]-1)/(X_train.shape[0]-X_train.shape[1]-1)
    # testing set scores
    y_pred = best_model.predict(X_test)
    RMSE_test = np.sqrt(mean_squared_error(Y_test,y_pred))
    R2_test = r2_score(Y_test,y_pred)
    adj_R2_test = 1-(1-R2_test)*(X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)
    

    print('Best model predictions\n Training Set\n RMSE = %.2f\t R2 = %.2f\t adj-R2 = %.2f\n Testing set\n RMSE = %.2f\t R2 = %.2f\t adj-R2=%.2f'
          %(RMSE_train,R2_train,adj_R2_train,RMSE_test,R2_test,adj_R2_test))
    
    return RMSE_train,R2_train,adj_R2_train,RMSE_test,R2_test,adj_R2_test


def model_fit(X_train,Y_train,cv,model):
    """
    
    Grid Search CV hyperparameters optimization

    Parameters
    ----------
    X_train : pandas DataFrame
            predictors training set
            
    Y_train : pandas DataFrame
            BC concentration training set
            
    cv : sklearn cv
    cross-validation scheme
    
    model : str
            ML algorithm for regression

    Returns
    -------
    gs : sklearn gridsearch object
        fitted grid for different hyperparameters combinations   
    
    results : pandas DataFrame
            gridsearch results for different hyperparamters combinations

    """
    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]
    # Select a model
    print('Hyperparameter\'s optimization for data set')
    print('%i predictors for BC: %s'%(n_features,[i for i in X_train.columns]))
    print('%i samples for training'%n_samples)
    
    if model=='SVR':
        print('---------------\nFitting SVR\n----------------')
        model = svm.SVR(kernel='rbf')
        grid_params = {
            'model__C':np.logspace(-3,3,7),
            'model__gamma':np.logspace(-3,3,7),
            'model__epsilon':np.linspace(0.1,0.8,5)
            }
        
        scaler = StandardScaler()
        pipe = Pipeline([('scaler',scaler),('model', model)])

    
    elif model=='RF':
        model = RandomForestRegressor(criterion='squared_error', min_samples_leaf=2,max_leaf_nodes=None,
                                      bootstrap=True, oob_score=False, random_state=92,
                                      n_jobs=2, verbose=1, warm_start=False)

        grid_params = {
            'model__n_estimators':[500,1000,3000],
            'model__max_depth':[20,10,7,5,3],
            'model__min_samples_split':[5,2],
            'model__max_samples':[1.0,0.66],
            'model__max_features':[1.0,0.5,0.33]
            }   
    
        pipe = Pipeline(steps=[('model', model)])


    elif model=='MLP':
        print('-----------\nFitting MLP\n----------')
        model = MLPRegressor(solver='adam',learning_rate='adaptive',
                             max_iter=5000,shuffle=True,random_state=92,warm_start=False)
        grid_params = {
            'model__activation':['tanh','relu'],
            'model__early_stopping':[False,True],
            'model__tol':np.logspace(-6,-2,3),
            'model__learning_rate_init':np.logspace(-4,-2,3),
            'model__hidden_layer_sizes':[(int(np.ceil(1.33*n_features)),),
                                         (n_features,),
                                         (int(0.5*n_features),),
                                         (n_features,n_features),
                                         (int(0.5*n_features),int(0.5*n_features)),
                                         (n_features,n_features,n_features,n_features,n_features)],
            'model__alpha':np.logspace(-6,-2,3),
            'model__batch_size': [int(0.01*n_samples),int(0.05*n_samples),int(0.1*n_samples),int(0.5*n_samples)]        
            }
        
        
        scaler = StandardScaler()
        pipe = Pipeline([('scaler',scaler),('model', model)])

    # gridsearch
    print('Fitting model ',pipe)
    start_time = time.time()
    
    if model =='RF':
        n_jobs = 2
        pre_dispatch = 2
    else:
        n_jobs = 2
        pre_dispatch = 4
        
    gs = GridSearchCV(pipe, grid_params, scoring='neg_root_mean_squared_error',
                      n_jobs=n_jobs, refit=True, cv=cv, verbose=1,
                      pre_dispatch=pre_dispatch, return_train_score=True)
    gs.fit(X_train,Y_train)
    end_time = time.time()
    gs_results = pd.DataFrame(gs.cv_results_)
    gs_results = gs_results.sort_values(by='rank_test_score')
    print('Grid search cv finished in %.2f'%(end_time-start_time))
    
    return gs,gs_results

        

def scoring(Y_true,y_pred,X):
    """
    Computes scoring metrics for predictions

    Parameters
    ----------
    Y_true : pandas series
            BC concentration ground truth value
    y_pred : pandas Series
            BC concentration estimation
    X : pandas DataFrame
        predictors training/or testing set

    Returns
    -------
    RMSE : float
            Root mean squared error
    R2 : float
        R2
    adj_R2 : float
            adjusted R2 (considers number of measurements and features)

    """
    
    RMSE = np.sqrt(mean_squared_error(Y_true,y_pred))
    R2 = r2_score(Y_true,y_pred)
    adj_R2 = 1-(1-R2)*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1)
    
    return RMSE,R2,adj_R2

#%%

def main(df):
    """
    Test hyperparamters optimization

    Parameters
    ----------
    df : pandas DataFrame
        data set with all predictors

    
    """
    X_tot = df.iloc[:,1:]
    Y_tot = df.iloc[:,0]
    test_frac = 0.25    
    X_train, X_test, Y_train, Y_test = train_test_split(X_tot,Y_tot,test_size=test_frac,
                                                        random_state=92,shuffle=True)
    cv = KFold(n_splits=10,shuffle=True,random_state=92)
    gs,results = model_fit(X_train,Y_train,cv,model='MLP')
    
    
    return gs,results,X_test,Y_test

#%%
if __name__ == '__main__':
    print('Testing model predictions')
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/1_Files/raw_data_files'
    Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = DS.load_or_create(path,load=True)
    ### -----------Pre-processing steps-----------
    dataSet = PP.pre_processing(Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N)
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/1_Files/raw_data_files/dataFrames_files'
    os.chdir(path)
    print('Loading Original data set from\n %s'%path)
    ds = pd.read_pickle('dataSet_original.pkl')
    # gridsearch
    gs,results,X_test,Y_test = main(ds)
    # scoring metrics
    y_pred = gs.predict(X_test)
    RMSE,R2,adj_R2 = scoring(Y_test,y_pred,X_test)
    # save results
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/3_BC_Proxy/Feature_Selection/MLP'
    os.chdir(path)
    print('Saving Results to %s'%path)   
    header = ['RMSE','R2','adj_R2']         
    data = [RMSE,R2,adj_R2]
    with open('MLP_gs_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(data)
    print('Saving model architecture')
    fname = 'BC_proxy_MLP_allFeatures.pkl'
    joblib.dump(gs, fname)

    
   