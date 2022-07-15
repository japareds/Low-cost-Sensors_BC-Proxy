#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBJECTIVE:
    This code aims to build a BC proxy from LCS datasets
    The methods used are SVR and RF

Created on Thu Jun 23 16:13:25 2022

@author: jparedes
"""
### system
import os
import sys
import warnings
import time
from datetime import datetime

### computational
import pandas as pd
import numpy as np
#from numpy import asarray, savetext, loadtxt

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split, KFold,cross_val_score, cross_validate,learning_curve, validation_curve,GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS 
import joblib
from tabulate import tabulate

### plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mtick
import seaborn as sns
from ipywidgets import *

### Modules
import BCP_DataSet as DS
import BCP_PreProcessing as PP
import BCP_Imputation as IMP
import BCP_LCS_Cal as CAL
import BCP_FeatureSelection as FS
import BCP_dataDriven_models as DDM

#%% functions
### imputation


### feature selection
def No_BFS(file,cv,X_train,Y_train):
    """
    Genreates BFS report in case it does not exist

    Returns
    -------
    None.
    Genrates BFS at path
    """
    if not os.path.exists(file):
        print('Feature Selection has not yet been carried out')
        BFS_results = FS.SVR_BFS(cv,X_train,Y_train,max_features=X_train.shape[1],min_features=1)
        FS.Explore_results(BFS_results, max_features=X_train.shape[1])
        return BFS_results
    else:
        print('Feature Selection done\n check results in')
        print(file)
        return 
    
### calibration
def calibrate_pollutants(df,pol_name):
    """
    Calibrates pollutants.
    There are 4 pollutants that have reference Station measurements: O3, NO2, NO, and PM10

    Parameters
    ----------
    df : pandas DataDrame
        data set containing all BC predictors no calibrated
    pol_name : str
        pollutant name for calibration

    Returns
    -------
            
    df_cal : pandas DataFrame
                data set with 2 new columns: pollutant calibrated via MLR and SVR
    
    results_MLR : pandas DataFrame
                    result of MLR grid search
    
    plot_results_MLR : matplotlib figure
                        plot of grid search MLR results            
    
    results_SVR : pandas DataFrame
                    result of SVR grid search
                    
    plot_results_SVR : matplotlib figure
                        plot of grid search SVR results
                        

    """
    ### calibrate pollutant
    print('calibrating %s'%pol_name)
    if pol_name == 'O3':
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/O3'
        os.chdir(path)
        print('Changing path to \n %s'%path)
        Y_target = df[['O3_Ref']]
        X_pol = df[['S_O3','T_int','RH_int']]
    elif pol_name == 'NO2':
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/NO2'
        os.chdir(path)
        print('Changing path to \n %s'%path)       
        Y_target = df[['NO2_Ref']]
        X_pol = df[['S_NO2','S_O3','T_int','RH_int']]
    elif pol_name == 'NO':
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/NO'
        os.chdir(path)
        print('Changing path to \n %s'%path)
        Y_target = df[['NO_Ref']]
        X_pol = df[['S_NO','S_O3','T_int','RH_int']]
    elif pol_name == 'PM10':
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/PM10'
        os.chdir(path)
        print('Changing path to \n %s'%path)
        Y_target = df[['PM10_Ref']]
        X_pol = df[['S5_PM10','T','RH']]
    else:
        print('Wrong pollutant name\n Available pollutants for calibration are\n O3, NO2, NO, PM10')
        return 
    
        
    test_frac= 0.25
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_pol,Y_target,test_size=test_frac,shuffle=True,random_state=92)
    cv = KFold(n_splits=10,shuffle=True,random_state=92)
    
    ### MLR
    #### search for model
    print('---------Calibrating using MLR---------')
    results_MLR = CAL.calibrate_MLR(X_train,Y_train,cv)
    plot_results_MLR = CAL.gs_results(results_MLR)
    #### select best model
    model_n = 0
    alpha = results_MLR.iloc[model_n]['param_model__alpha']
    final_model = Ridge(fit_intercept=True,tol=1e-4,solver='auto',alpha=alpha)
    final_pipe = Pipeline(steps=[('model',final_model)])
    print('Final model chosen')
    print(final_pipe)
    final_pipe.fit(X_train,Y_train.values.ravel())
    RMSE_train,R2_train,RMSE_test,R2_test = CAL.metrics(X_train,Y_train,X_test,Y_test,final_pipe)
    print('Scoring results:\n')
    print('Training set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_train,R2_train))
    print('Testing set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_test,R2_test))
    ### calibrated pollutant
    y_cal_MLR = final_pipe.predict(X_pol)
    y_cal_MLR[y_cal_MLR<0.]=0.
    y_cal_MLR = pd.DataFrame(y_cal_MLR,columns=[pol_name+'_cal_MLR'],index=X_pol.index)
    ### save results
    print('Saving MLR results')
    #fname = pol_name+'_gridSearch_results_MLR.pkl'
    #results_MLR.to_pickle(fname)
    #fname = pol_name+'_calibration_model_MLR.pkl'
    #joblib.dump(final_pipe, fname)
    #fname = pol_name+'_MLR_calibrated_dataSet.pkl'
    #y_cal_MLR.to_pickle(fname)
    fname = pol_name+'_MLR_calibration_scores.txt'
    f = open(fname,'a')
    print('-------------',file=f)
    print(datetime.now(),file=f)
    print(pol_name+' calibration',file=f)
    print('Algorithm MLR',file=f)
    print('Model: ',final_pipe,file=f)
    print('Scoring results:\n',file=f)
    print('Training set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_train,R2_train),file=f)
    print('Testing set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_test,R2_test),file=f)
    
    ### SVR
    print('---------Calibrating using SVR---------')
    #### model search
    results_SVR = CAL.calibrate_SVR(X_train,Y_train,cv)
    plot_results_SVR = CAL.gs_results(results_SVR)
    #### model selection
    model_n = 0
    C = results_SVR.iloc[model_n]['param_model__C']
    g = results_SVR.iloc[model_n]['param_model__gamma']
    e = results_SVR.iloc[model_n]['param_model__epsilon']
    final_model = svm.SVR(kernel='rbf',C=C,gamma=g,epsilon=e)
    scaler = StandardScaler()
    final_pipe = Pipeline(steps=[("scaler", scaler), ("model", final_model)]                   )
    print('Final model chosen')
    print(final_pipe)
    final_pipe.fit(X_train,Y_train.values.ravel())
    RMSE_train,R2_train,RMSE_test,R2_test = CAL.metrics(X_train,Y_train,X_test,Y_test,final_pipe)
    print('Scoring results:\n')
    print('Training set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_train,R2_train))
    print('Testing set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_test,R2_test))
    #### calibrated pollutant 
    y_cal_SVR = final_pipe.predict(X_pol)
    y_cal_SVR[y_cal_SVR<0.]=0.
    y_cal_SVR = pd.DataFrame(y_cal_SVR,columns=[pol_name+'_cal_SVR'],index=X_pol.index)
    ### save results
    print('Saving MLR results')
    #fname = pol_name+'_gridSearch_results_SVR.pkl'
    #results_MLR.to_pickle(fname)
    #fname = pol_name+'_calibration_model_SVR.pkl'
    #joblib.dump(final_pipe, fname)
    #fname = pol_name+'_SVR_calibrated_dataSet.pkl'
    #y_cal_MLR.to_pickle(fname)
    fname = pol_name+'_SVR_calibration_scores.txt'
    f = open(fname,'a')
    print('-------------',file=f)
    print(datetime.now(),file=f)
    print(pol_name+' calibration',file=f)
    print('Algorithm MLR',file=f)
    print('Model: ',final_pipe,file=f)
    print('Scoring results:\n',file=f)
    print('Training set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_train,R2_train),file=f)
    print('Testing set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_test,R2_test),file=f)

    ### add calibrated pollutant to data set
    df_cal = pd.concat([df,y_cal_MLR,y_cal_SVR],axis=1)
    
    return df_cal,results_MLR,plot_results_MLR,results_SVR,plot_results_SVR

def calibrated_models_load(df,pol_name):
    """
    Load calibration models found previously via grid search

    Parameters
    ----------
    df : pandas DataFrame
        data set of BC predictors
        
    pol_name : str
        name of pollutant for calibration

    Returns
    -------
    df_cal : padnas DataFrame
        data set with calibrated pollutant appended

    """
    print('calibrating %s'%pol_name)
    if pol_name == 'O3':
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/O3'
        os.chdir(path)
        print('Changing path to \n %s'%path)
        Y_target = df[['O3_Ref']]
        X_pol = df[['S_O3','T_int','RH_int']]
        model_MLR = joblib.load('O3_calibration_model_MLR.pkl')
        model_SVR = joblib.load('O3_calibration_model_SVR.pkl')
    elif pol_name == 'NO2':
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/NO2'
        os.chdir(path)
        print('Changing path to \n %s'%path)       
        Y_target = df[['NO2_Ref']]
        X_pol = df[['S_NO2','S_O3','T_int','RH_int']]
        model_MLR = joblib.load('NO2_calibration_model_MLR.pkl')
        model_SVR = joblib.load('NO2_calibration_model_SVR.pkl')        
    elif pol_name == 'NO':
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/NO'
        os.chdir(path)
        print('Changing path to \n %s'%path)
        Y_target = df[['NO_Ref']]
        X_pol = df[['S_NO','S_O3','T_int','RH_int']]
        model_MLR = joblib.load('NO_calibration_model_MLR.pkl')
        model_SVR = joblib.load('NO_calibration_model_SVR.pkl')

    elif pol_name == 'PM10':
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/PM10'
        os.chdir(path)
        print('Changing path to \n %s'%path)
        Y_target = df[['PM10_Ref']]
        X_pol = df[['S5_PM10','T','RH']]
        model_MLR = joblib.load('PM10_calibration_model_MLR.pkl')
        model_SVR = joblib.load('PM10_calibration_model_SVR.pkl')

    else:
        print('Wrong pollutant name\n Available pollutants for calibration are\n O3, NO2, NO, PM10')
        return
    # Fitting models
    test_frac= 0.25
    X_train, X_test, Y_train, Y_test = train_test_split(X_pol,Y_target,test_size=test_frac,shuffle=True,random_state=92)
    model_MLR.fit(X_train,Y_train.values.ravel())
    model_SVR.fit(X_train,Y_train.values.ravel())
    # Calibrated pollutant
    ## MLR
    y_cal_MLR = model_MLR.predict(X_pol)
    y_cal_MLR[y_cal_MLR<0.]=0.
    y_cal_MLR = pd.DataFrame(y_cal_MLR,columns=[pol_name+'_cal_MLR'],index=X_pol.index)
    ## SVR
    y_cal_SVR = model_SVR.predict(X_pol)
    y_cal_SVR[y_cal_SVR<0.]=0.
    y_cal_SVR = pd.DataFrame(y_cal_SVR,columns=[pol_name+'_cal_SVR'],index=X_pol.index)
    
    # Add calibrated pollutants to data set
    df_cal = pd.concat([df,y_cal_MLR,y_cal_SVR],axis=1)
    
    return df_cal

def pre_imputation(dataSet):
    """
    pre-Proxy imputation

    Parameters
    ----------
    dataSet : pandas DataFrame
            predictors and BC data set

    Returns
    -------
    imp_X_train : pandas DataFrame
                predictors training set with imputed values
                
    imp_X_test : pandas DataFrame
                predictors testing set with imputed values
                
    imp_Y_train : pandas DataFrame
                BC training set with imputed values
                
    imp_Y_test : pandas DataFrame
                BC testing set with imputed values

    """
    Y_tot = dataSet.iloc[:,0]
    X_tot = dataSet.iloc[:,1:]
    X_train, X_test, Y_train, Y_test = train_test_split(X_tot,Y_tot,test_size=0.25,shuffle=True,random_state=92)    
    print('---------------\nImputation techniques\n---------------')
    time.sleep(2)
    print('Creating imputer from training set')

    #### pollutants to impute
    pollutants = ['O3','NO2','NO','PM10','N_Ref','T','RH','O3','NO2','NO','T','RH','PM1','PM25','PM10','N0.3','N0.5','N1.0','N2.5','N5.0','N10.0']
    ref = [True,True,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False]
    imp_pollutants = []
    imputers = []

    #### imputation method
    method = 'MICE'
    for p,r in zip(pollutants,ref):
        df = IMP.variables(X_train,imp_pollutant=p,ref=r)
        imp_pol,imp = IMP.imputation(df, method=method,is_train=True)
    
    ### save pollutant with imputed values
    pollutant_imp = pd.DataFrame(imp_pol.iloc[:,0])
    imp_pollutants.append(pollutant_imp)
    imputers.append(imp)
    
    #### joint imputed pollutants into one data frame
    imp_X_train = pd.DataFrame()
    for i in range(len(imp_pollutants)):
        imp_X_train[imp_pollutants[i].columns[0]] = imp_pollutants[i].iloc[:,0]

    #### apply the imputers to the testing set
    print('Transforming testing set')
    imp_pollutants = []
    idx = [i for i in range(0,len(ref))]

    #### imputation of testing set
    for p,r,i in zip(pollutants,ref,idx):
        df = IMP.variables(X_test,imp_pollutant=p,ref=r)
    
    imp_pol,imp = IMP.imputation(df, method=method,is_train=False,imp=imputers[i])
    
    ### save pollutant with imputed values
    pollutant_imp = pd.DataFrame(imp_pol.iloc[:,0])
    imp_pollutants.append(pollutant_imp)
    imputers.append(imp)
    
    #### updated X_test with imputed values
    imp_X_test = pd.DataFrame()
    for i in range(len(imp_pollutants)):
        imp_X_test[imp_pollutants[i].columns[0]] = imp_pollutants[i].iloc[:,0]

    ### imputation of BC
    imp_Y_train,BC_median =  IMP.impute_target(Y_train,is_train=True,value=None)
    imp_Y_test,BC_median =  IMP.impute_target(Y_test,is_train=False,value=BC_median)
    
    print('----------------\nImputation Finished\n--------------')
    time.sleep(1.5)
    
    return imp_X_train,imp_Y_train,imp_Y_train,imp_Y_test

def models_prediction(data):
    """
    Predict BC concentration using an specific ML model:
        - SVR
        - RF
        - MLP
    

    Parameters
    ----------
    data : pandas DataFrame
        data set containing BC and predictors

    Returns
    -------
    data : pandas DataFrame
            data set containing BC and predictors
    X_train : pandas DataFrame
            predictors training set
            
    X_test : pandas DataFrame
            predictors testing set
            
    Y_train : pandas DataFrame
            BC concentration training set
            
    Y_test : pandas DataFrame
            BC concentration testing set
            
    gs : sklearn grid search
        grid search fitted
        
    results : pandas DataFrame
        grid search cv results

    """
    
    print('Machine Learning model hyperparamters optimization')
    time.sleep(2)
    #path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/3_BC_Proxy/Feature_Selection/MLP'
    #os.chdir(path)
    #print('changing directory: %s'%path)
    optimal_subset = False
    if optimal_subset:
        var = ['BC','O3_cal_SVR','NO2_cal_SVR','S5_0.3','S5_1.0','T','RH']
        df = data[var]
        print('Fitting using optimal subset as predictors ')
        print('%s'%[i for i in df.columns])
    else:
        df = data.copy()
        
    X_tot = df.iloc[:,1:]
    Y_tot = df.iloc[:,0]
    test_frac = 0.25    
    X_train, X_test, Y_train, Y_test = train_test_split(X_tot,Y_tot,test_size=test_frac,
                                                        random_state=92,shuffle=True)
    cv = KFold(n_splits=10,shuffle=True,random_state=92)
    model = 'SVR'
    gs,results = DDM.model_fit(X_train,Y_train,cv,model=model)
    print('Saving results')
    fname = 'BC_Proxy_LCS_'+model+'_Subset_5Predictors_No1.0.pkl'
    results.to_pickle(fname)
    
    RMSE_train,R2_train,adj_R2_train,RMSE_test,R2_test,adj_R2_test = DDM.best_estimator_predict(gs,X_train,X_test,Y_train,Y_test)
    fname = 'report_'+model+'_GS_metrics.txt'
    f = open(fname,'a')
    print('--------------------',file=f)
    print(model+' results',file=f)
    print('Predictors: ',file=f)
    print([i for i in X_test.columns],file=f)
    print('Best model found via GS: ',file=f)
    print(gs.best_estimator_,file=f)
    print('Training set\n RMSE = %.2f\n R2=%.2f\n adj-R2=%.2f'%(RMSE_train,R2_train,adj_R2_train),file=f)
    print('Testing set\n RMSE = %.2f\n R2=%.2f\n adj-R2=%.2f'%(RMSE_test,R2_test,adj_R2_test),file=f)
    f.close()
    
    return data,X_train,X_test,Y_train,Y_test,gs,results

def model_deploy(df,method='SVR'):
    """
    BC proxy model deployment
    The models weere previously found via GridSearch hyperaparameters optimization.
    
    Possible models: SVR,RF,MLP

    Parameters
    ----------
    df : pandas DataFrame
        data set of Bc and predictors
        
    method : str
        ML method: SWVR, RF, MLP

    Returns
    -------
    X_train : pandas DataFrame
            predictors training set
            
    X_test : pandas DataFrame
            predictors testing set
            
    Y_train : pandas DataFrame
            BC training set
            
    Y_test : pandas DataFrame
            BC testing set
            
    pipe : sklearn pipeline
        model and pre steps if necessary

    """
    # train-test split
    X_tot = df.iloc[:,1:]
    Y_tot = df.iloc[:,0]
    test_frac = 0.25    
    X_train, X_test, Y_train, Y_test = train_test_split(X_tot,Y_tot,test_size=test_frac,
                                                        random_state=92,shuffle=True)
    # model selection
    if method=='SVR':
        model = svm.SVR(C=1, epsilon=0.2, gamma=1)
        scaler = StandardScaler()
        pipe = Pipeline([('scaler',scaler),('model', model)])
        
        
    print('Model deployed')
    print(pipe)    
    # fit model on training set
    pipe.fit(X_train,Y_train)
    return X_train,X_test,Y_train,Y_test,pipe

def scoring(Y_true,y_pred,X):
    
    RMSE = np.sqrt(mean_squared_error(Y_true,y_pred))
    R2 = r2_score(Y_true,y_pred)
    adj_R2 = 1-(1-R2)*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1)
    
    return RMSE,R2,adj_R2
#%%
###############################################################################
#
#       Functions on missings imputation during deployment
#
###############################################################################

def generate_missings(df,fraction=0.5,missing_seed = 0):
    """
    Generate missings on data all columns of data frame (df) based on fraction

    Parameters
    ----------
    df : pandas DataFrame
        data set with different predictors as columns
        
    fraction : float
        fraction of points to be set as nan. Default is 50%
        
    missing_seed : int
                random state for location of missings

    Returns
    -------
    df_ : pandas DataFrame
        data set with nan entries at locations

    """
    print('Generating missings on %.1f percent of data'%(fraction*100))
    # sample randomly from data frame and create missing values
    df_sample = df.sample(frac=fraction,replace=False,random_state=missing_seed,axis=0)
    idx = df_sample.index
    df_ = df.copy()
    df_.loc[idx] = np.nan   

    return df_

def nan_imputation(df,method='MICE'):
    """
    Simple nan imputation according to method

    Parameters
    ----------
    df : pandas DataFrame
        data set with missing values as nan
        
    method : str
        method for imputation (MICE, MissForest,KNN)

    Returns
    -------
    imp_df : pandas DataFrame
        data set with imputed data
        
    imp : sklearn imputer
        imputer object fitted

    """
    imp_df,imp = IMP.imputation(df, method,is_train=True,imp=None,n_iter=30)   
    return imp_df,imp

def deployment_missings(Y_test,X_test,pipe):
    """
    Creates missings at random positions and imputes them for evaluating
    BC proxy performance
    

    Parameters
    ----------
    Y_test: pandas DataFrame
            BC concetration testing set
            
    X_test : pandas DataFrame
            predictors testing set
            
    pipe : sklearn pipe
            pipeline with model at last step

    Returns
    -------
    RMSE_pred : list
            RMSE for every predictor for different proportion of missings 
            and different imputation methods
            
    R2_pred : list
        R2 for every predictor for different proportion of missings 
        and different imputation methods
    
    adj_R2_pred : list
        adjusted R2 for every predictor for different proportion of missings 
        and different imputation methods

    """
    predictors = [i for i in X_test.columns]
    # for every predictor...
    RMSE_pred = []
    R2_pred = []
    adj_R2_pred = []
    
    single_predictor = False
    
    # missingess for 1 single predictor
    if single_predictor:
        print('\n----------------\nGenerating missings for individual predictors\n-------------')
        time.sleep(2)
        for p in predictors:
            print('\n---------------\nGenerating missings on predictor %s\n----------------'%p)
            # generate different fractions of empty values...
            RMSE_missings = []
            R2_missings = []
            adj_R2_missings = []
            for i in np.linspace(0.05,0.5,10):
                predictor_missings = generate_missings(X_test[[p]],fraction=i)
                predictor_missings = pd.DataFrame(predictor_missings)
                predictor_missings.columns = predictor_missings.columns+'_missing_'+str(i)
                X_test_missings = pd.concat([predictor_missings,X_test],axis=1)
                X_test_missings = X_test_missings.drop(labels=p,axis=1)
                # which are imputated with different methods
                RMSE_method = []
                R2_method = []
                adj_R2_method = []
                for m in ['MICE','MissForest']:
                    predictor_imputed,imp = nan_imputation(X_test_missings,method=m)
                    predictor_imputed = pd.DataFrame(predictor_imputed.iloc[:,0])
                    predictor_imputed.columns = predictor_imputed.columns+'_imp_'+m
                    X_test_imputed = pd.concat([predictor_imputed,X_test],axis=1)
                    X_test_imputed = X_test_imputed.drop(labels=p,axis=1)
                    
                    # predict with imputed data set
                    X_test_pred = X_test_imputed.copy()
                    X_test_pred.rename(columns={X_test_pred.columns[0]: p},inplace=True)
                    y_pred = pipe.predict(X_test_pred[X_test.columns])
                    
                    # metrics
                    RMSE,R2,adj_R2 = scoring(Y_test,y_pred,X_test)
                    # save metrics for imputation strategy
                    RMSE_method.append(RMSE)
                    R2_method.append(R2)
                    adj_R2_method.append(adj_R2)
                    
                    # save metrics for missings proportion
                    RMSE_missings.append(RMSE_method)
                    R2_missings.append(R2_method)
                    adj_R2_missings.append(adj_R2_method)
                    
        
            # save metrics for predictor missing
            RMSE_pred.append(RMSE_missings)
            R2_pred.append(R2_missings)
            adj_R2_pred.append(adj_R2_missings)
            
            df_RMSE = pd.DataFrame(RMSE_missings)
            df_RMSE.columns = ['MICE','MissForest']
            df_RMSE.index = [i for i in np.linspace(0.05,0.5,10)]
            fname = 'RMSE_imp_'+p+'_s9.pkl'
            df_RMSE.to_pickle(fname)
        
            df_R2 = pd.DataFrame(R2_missings)
            df_R2.columns = ['MICE','MissForest']
            df_R2.index = [i for i in np.linspace(0.05,0.5,10)]
            fname = 'R2_imp_'+p+'_s9.pkl'
            df_R2.to_pickle(fname)
        
            df_adj_R2 = pd.DataFrame(adj_R2_missings)
            df_adj_R2.columns = ['MICE','MissForest']
            df_adj_R2.index = [i for i in np.linspace(0.05,0.5,10)]
            fname = 'adjR2_imp_'+p+'_s9.pkl'
            df_adj_R2.to_pickle(fname)
        
        
    # missingess for multiple predictors
    else: 
        print('\n---------------\nGenerating missings on all predictors simultaneously\n----------------')
        time.sleep(2)
        RMSE_missings = []
        R2_missings = []
        adj_R2_missings = []
        # generate different fractions of empty values
        for i in np.linspace(0.05,0.5,10):
            X_test_missings_ = pd.DataFrame()
            # generate the data set with empty entries
            for p in predictors:
                print('Generating missing entries on predictor %s'%p)
                # generate missing entries but not the same missing entries for ALL the predictors
                predictor_missings = generate_missings(X_test[[p]],fraction=i,missing_seed = 0+[i for i in X_test.columns].index(p))
                predictor_missings = pd.DataFrame(predictor_missings)
                predictor_missings.columns = predictor_missings.columns+'_missing_'+str(i)
                X_test_missings = pd.concat([predictor_missings,X_test],axis=1)
                X_test_missings = X_test_missings.drop(labels=p,axis=1)
                X_test_missings_ = pd.concat([X_test_missings_,X_test_missings.iloc[:,0]],axis=1)
            
            # imputation of empty values            
            RMSE_method = []
            R2_method = []
            adj_R2_method = []
            for m in ['MICE','MissForest']:
                predictor_imputed,imp = nan_imputation(X_test_missings_,method=m)
                predictor_imputed = pd.DataFrame(predictor_imputed)
                predictor_imputed.columns = predictor_imputed.columns+'_imp_'+m
                X_test_imputed = predictor_imputed.copy()
                
                # predict with imputed data set
                X_test_pred = X_test_imputed.copy()
                for p in range(len(predictors)):
                    X_test_pred.rename(columns={X_test_pred.columns[p]: predictors[p]},inplace=True)
                    
                y_pred = pipe.predict(X_test_pred[X_test.columns])
                
                # metrics
                RMSE,R2,adj_R2 = scoring(Y_test,y_pred,X_test)
                # save metrics for imputation strategy
                RMSE_method.append(RMSE)
                R2_method.append(R2)
                adj_R2_method.append(adj_R2)
                
            # save metrics for missings proportion
            RMSE_missings.append(RMSE_method)
            R2_missings.append(R2_method)
            adj_R2_missings.append(adj_R2_method)
                
            
            
            
        # save metrics for missing predictor
        RMSE_pred = RMSE_missings
        R2_pred = R2_missings
        adj_R2_pred = adj_R2_missings
        
        df_RMSE = pd.DataFrame(RMSE_missings)
        df_RMSE.columns = ['MICE','MissForest']
        df_RMSE.index = [i for i in np.linspace(0.05,0.5,10)]
        fname = 'RMSE_imp_multipleMissings_s0.pkl'
        df_RMSE.to_pickle(fname)
    
        df_R2 = pd.DataFrame(R2_missings)
        df_R2.columns = ['MICE','MissForest']
        df_R2.index = [i for i in np.linspace(0.05,0.5,10)]
        fname = 'R2_imp_multipleMissings_s0.pkl'
        df_R2.to_pickle(fname)
    
        df_adj_R2 = pd.DataFrame(adj_R2_missings)
        df_adj_R2.columns = ['MICE','MissForest']
        df_adj_R2.index = [i for i in np.linspace(0.05,0.5,10)]
        fname = 'adjR2_imp_multipleMissings_s0.pkl'
        df_adj_R2.to_pickle(fname)
        
        
    return RMSE_pred,R2_pred,adj_R2_pred
    
                

#%%

def main():
    """
    main

    RETURNS 
    
    dataSet: data frame with BC and rpedictors every time
    BFS_results: feature selection via wrapper method
    """
    ########################################
    #----------- load dataset-----------
    # load all predictors
    # pre-processing step
    ########################################

    ### -----------load dataset-----------
    """
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/1_Files/raw_data_files'
    Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = DS.load_or_create(path,load=True)
    ### -----------Pre-processing steps-----------
    dataSet = PP.pre_processing(Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N)
    """
    # original data set
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/1_Files/raw_data_files/dataFrames_files'
    os.chdir(path)
    print('Loading Original data set from\n %s'%path)
    ds = pd.read_pickle('dataSet_original.pkl')
    ########################################
    #----------- LCS calibration-----------
    # model selection takes time - try to do it only once
    # cal: bool - search for models or skip
    ########################################
    """
    cal = False
    if cal:
        print('-------------\nCalibrating LCS\n-------------\n')
        time.sleep(2)
        df_cal_O3,results_MLR_O3,plot_results_MLR_O3,results_SVR_O3,plot_results_SVR_O3 = calibrate_pollutants(imp_X_train,'O3')
        df_cal_NO2,results_MLR_NO2,plot_results_MLR_NO2,results_SVR_NO2,plot_results_SVR_NO2 = calibrate_pollutants(df_cal_O3,'NO2')
        df_cal_NO,results_MLR_NO,plot_results_MLR_NO,results_SVR_NO,plot_results_SVR_NO = calibrate_pollutants(df_cal_NO2,'NO')
        df_cal_PM10,results_MLR_PM10,plot_results_MLR_PM10,results_SVR_PM10,plot_results_SVR_PM10 = calibrate_pollutants(df_cal_NO,'PM10')
        ds = df_cal_PM10.copy()

    else:
        print('--------------\nLoading Calibration models\n-------------- ')
        time.sleep(2)
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/O3'
        os.chdir(path)
        print('Changing directory: %s'%path)
        df_cal_O3 = pd.read_pickle('O3_cal_SVR.pkl')
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/NO2'
        os.chdir(path)
        print('Changing directory: %s'%path)
        df_cal_NO2 = pd.read_pickle('NO2_cal_SVR.pkl')
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/NO'
        os.chdir(path)
        print('Changing directory: %s'%path)
        df_cal_NO = pd.read_pickle('NO_cal_SVR.pkl')
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/PM10'
        os.chdir(path)
        print('Changing directory: %s'%path)
        df_cal_PM10 = pd.read_pickle('PM10_cal_SVR.pkl')
        
    variables = ['BC','T','RH','S5_PM1','S5_PM25','S5_PM10','S5_0.3','S5_0.5','S5_1.0','S5_2.5','S5_2.5','S5_5.0','S5_10.0']
    df = dataSet[variables]
    print('-------------\nCreating LCS dataset for proxy\n---------------')
    data = pd.concat([df,df_cal_O3,df_cal_NO2,df_cal_NO,df_cal_PM10],axis=1)
    data.dropna(inplace=True)    
    print('Number of samples %i'%data.shape[0])
    """
   
    
    ###########################################################################
    #---------------- Feature Selection -----------------
    # Backwards feature selection
    # find: bool - search for model or skip and load an already found model
    ###########################################################################
    feature_selection= False
    
    """  
        
        
        #### perform BFS report in case it does not exist
        file = "/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/3_BC_Proxy/Feature_Selection/BFS/Report_SVR_BFS.txt"
        No_BFS(file,cv,X_train,Y_train)
    """

    ###########################################################################
    #---------------- proxy model prediction -----------------
    # model hyperparameters optimization
    # predict using features found via BFS
    ###########################################################################
    model_pred = False
    if model_pred:
        data,X_train,X_test,Y_train,Y_test,gs,results = models_prediction(ds)
    
    ###########################################################################
    #------------ Model deployment------------------
    # Select a model from the previously found via grid search
    #
    ###########################################################################
    deployment = True
    if deployment:
        # predictors
        var = ['BC','O3_cal_SVR','NO2_cal_SVR','S5_0.3','S5_1.0','T','RH']
        ds = ds[var]
        print('Fitting using optimal subset as predictors ')
        print('%s'%[i for i in ds.columns])
        # model deployment
        X_train,X_test,Y_train,Y_test,pipe = model_deploy(ds,method='SVR')
    ###########################################################################
    #-------------------------- Proxy model robustness-------------------------
    # missing values imputation on predictors
    ###########################################################################
    missings = True
    if missings:
        print('------------------\nSimulating missing values during deployment\n------------------')
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/4_Proxy_deployment/Missing_values/Imputation'
        os.chdir(path)
        print('Changing directory\n%s'%path)
        RMSE_pred,R2_pred,adj_R2_pred = deployment_missings(Y_test,X_test,pipe)
        
     
    lacking_sensor = False
    if lacking_sensor:
        # optimal subset found
        
        var = ['BC','O3_cal_SVR','NO2_cal_SVR','S5_0.3','S5_1.0','T','RH']
        ds = ds[var]

        print('---------------\n Simulating lacking a sensor during deployment\n------------------')
        time.sleep(2)
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/4_Proxy_deployment/Missing_values/Regression_remaining'
        os.chdir(path)
        print('Changing directory\n%s'%path)
        
        # dropping 1 feature 
        drop = 'S5_1.0'
        ds.drop(drop,axis=1,inplace=True)
        print('Lacking sensor for %s'%drop)
        print('Remaining sensors: ')
        print([i for i in ds.columns[1:]])
        
        # fitting with the rest
        data,X_train,X_test,Y_train,Y_test,gs,results = models_prediction(ds)
        
        
        
        
       
        
    return ds,X_train,X_test,Y_train,Y_test
    


if __name__ == '__main__':
    ds,X_train,X_test,Y_train,Y_test = main()
    print('--------\nDone\n---------')
    

