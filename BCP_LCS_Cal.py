#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBJECTIVE:
    Calibrate DataSets from LCS 
    using reference Station measurements       
    
Receives dataset already pre-processed and returns 
updated dataset including columns with calibrated measurements
    
Created on Wed Jun 29 13:22:22 2022

@author: jparedes
"""
import numpy as np
import pandas as pd
from datetime import datetime
import os
### ML
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn import svm
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
### plot
import matplotlib.pyplot as plt

### Modules
import BCP_DataSet as DS
import BCP_PreProcessing as PP
#%%
def calibration_datasets(Ref_O3,Ref_NO2,Ref_NO,Ref_PM10,Ref_Meteo,LCS_O3,LCS_NO2,LCS_NO,LCS_PM10,LCS_Meteo,cal_pol='O3',include_ref_station=True):
    """
    Create calibration data sets for the ML model
    The reference station measurements are located in the first column

    Parameters
    ----------
    Ref_O3 : TYPE
        DESCRIPTION.
    Ref_NO2 : TYPE
        DESCRIPTION.
    Ref_NO : TYPE
        DESCRIPTION.
    Ref_PM10 : TYPE
        DESCRIPTION.
    Ref_Meteo : TYPE
        DESCRIPTION.
    LCS_O3 : TYPE
        DESCRIPTION.
    LCS_NO2 : TYPE
        DESCRIPTION.
    LCS_NO : TYPE
        DESCRIPTION.
    LCS_Meteo : TYPE
        DESCRIPTION.
    cal_pol : str
        pollutant to be calibrated
    include_ref_station : bool
        include reference data set into calibration data set, useful for training the calibration model. default is True

    Returns
    -------
    df : pandas DataFrame
        processed data set ready for calibration

    """
    time_period = '10min'
    if cal_pol == 'O3':
        # time aggregation
        Ref_O3_ = Ref_O3.groupby(pd.Grouper(key='date',freq=time_period)).mean()
        LCS_O3_ = LCS_O3.loc[:,['date','S_O3']].groupby(pd.Grouper(key='date',freq=time_period)).mean()
        LCS_Meteo_ = LCS_Meteo.groupby(pd.Grouper(key='date',freq=time_period)).mean()
        # df creation
        if include_ref_station:
            df = pd.concat([Ref_O3_,LCS_O3_,LCS_Meteo_],axis=1)
        else:
            df = pd.concat([LCS_O3_,LCS_Meteo_],axis=1)
    elif cal_pol == 'NO2':
        # time aggregation
        Ref_NO2_ = Ref_NO2.groupby(pd.Grouper(key='date',freq=time_period)).mean()
        LCS_NO2_ = LCS_NO2.loc[:,['date','S_NO2']].groupby(pd.Grouper(key='date',freq=time_period)).mean()
        LCS_O3_ = LCS_O3.loc[:,['date','S_O3']].groupby(pd.Grouper(key='date',freq=time_period)).mean()
        LCS_Meteo_ = LCS_Meteo.groupby(pd.Grouper(key='date',freq=time_period)).mean()
        # df creation
        if include_ref_station:
            df = pd.concat([Ref_NO2_,LCS_NO2_,LCS_O3_,LCS_Meteo_],axis=1)
        else:
            df = pd.concat([LCS_NO2_,LCS_O3_,LCS_Meteo_],axis=1)
    elif cal_pol == 'NO':
        # time aggregation
        Ref_NO_ = Ref_NO.groupby(pd.Grouper(key='date',freq=time_period)).mean()
        LCS_NO_ = LCS_NO.loc[:,['date','S_NO']].groupby(pd.Grouper(key='date',freq=time_period)).mean()
        LCS_O3_ = LCS_O3.loc[:,['date','S_O3']].groupby(pd.Grouper(key='date',freq=time_period)).mean()
        LCS_Meteo_ = LCS_Meteo.groupby(pd.Grouper(key='date',freq=time_period)).mean()
        # df creation
        if include_ref_station:
            df = pd.concat([Ref_NO_,LCS_NO_,LCS_O3_,LCS_Meteo_],axis=1)
        else:
            df = pd.concat([LCS_NO_,LCS_O3_,LCS_Meteo_],axis=1)
    elif cal_pol == 'PM10':
        # time aggregation
        Ref_PM10_ = Ref_PM10.groupby(pd.Grouper(key='date',freq=time_period)).mean()
        LCS_PM10_ = LCS_PM10.groupby(pd.Grouper(key='date',freq=time_period)).mean()
        Ref_Meteo_ = Ref_Meteo.groupby(pd.Grouper(key='date',freq=time_period)).mean()
        # df creation
        if include_ref_station:
            df = pd.concat([Ref_PM10_,LCS_PM10_,Ref_Meteo_],axis=1)
        else:
            df = pd.concat([LCS_PM10_,Ref_Meteo_],axis=1)
    else:
        print('No reference measurements available for selected pollutant')
        df = pd.DataFrame()
        
    # clean data set
    df.dropna(inplace=True)
    
    return df
#%%
def calibration_train_test_split(ds,test_frac=0.25):
    """
    Split the data set into training and testing sets
    The data set must be organized such that the first column corresponds to the BC concentration, to be estimated
    The data set is shuffled before partiotioning to ensure the BC proxy model is challenged with a wide variety of data

    Parameters
    ----------
    ds : pandas DataFrame
        data set containing all the data
    test_frac : int
        fraction of measurements for the testing set, defaulñt is 25%

    Returns
    -------
    X_train : pandas DataFrame
            BC proxy predictors training set
    Y_train : pandas DataFrame
            BC measurements training set
    X_test : pandas DataFrame
            BC proxy predictors testing set
    Y_test : pandas DataFrame
            BC measurements testing set

    """
    print('Splitting data set into training and testing set')
    X_tot = ds.iloc[:,1:]
    Y_tot = ds.iloc[:,0]
    X_train, X_test, Y_train, Y_test = train_test_split(X_tot,Y_tot,test_size=test_frac,random_state=92,shuffle=True)
    print('Training data set contains %i measurements for the %i predictors'%(X_train.shape[0],X_train.shape[1]))
    print('Predictors: %s'%[i for i in X_train.columns])
    print('Testing data set contains %i measurements for the %i predictors'%(X_test.shape[0],X_test.shape[1]))
    print('Predictors: %s'%[i for i in X_test.columns])
    
    
    return X_train,Y_train,X_test,Y_test

#%% calibration methods

### calibrate using MLR
def calibrate_MLR(X_train,Y_train,cv):
    """
    Calibrate LCS measurements using MLR

    Parameters
    ----------
    X_train : pandas DataFrame
        LCS data set with predictors
    Y_train : pandsa DataFrame
        Reference Station data set of target pollutant to be callibrated
    cv : sklearn KFOLD
        folds for k fold cross validation 

    Returns
    -------
    results : pandas DataFrame
            grid search results for all combination of hyperparameters

    """

    ### model
    model = Ridge(fit_intercept=True,tol=1e-4,solver='auto')
    pipe = Pipeline(steps=[('model',model)])
    grid_params = {
        'model__alpha': np.linspace(0,10,20),
    }
    
    ### gridsearch
    gs = GridSearchCV(pipe, grid_params, scoring = 'neg_root_mean_squared_error',
                        cv=cv,n_jobs=10,refit=True,verbose=1,
                        pre_dispatch=20,return_train_score=True)
    gs.fit(X_train,Y_train.values.ravel())     

    results = pd.DataFrame(gs.cv_results_)
    results = results.sort_values(by='rank_test_score')
    
    return results

### calibrate using SVR
def calibrate_SVR(X_train,Y_train,cv):
    """
    Calibrate LCS measurements using SVR

    Parameters
    ----------
    X_train : pandas DataFrame
        LCS data set with predictors
    Y_train : pandsa DataFrame
        Reference Station data set of target pollutant to be callibrated
    cv : sklearn KFOLD
        folds for k fold cross validation 

    Returns
    -------
    results : pandas DataFrame
            grid search results for all combination of hyperparameters

    """
    ### model
    scaler = StandardScaler()
    model = svm.SVR(kernel='rbf')
    pipe = Pipeline(steps=[('scaler',scaler),('model',model)])
    ### gridsearch
    grid_params = {
        'model__C': np.logspace(-1,2,7),
        'model__gamma': np.logspace(-2,1,6),
        'model__epsilon': np.linspace(0.1,1.0,4)
        }

    gs = GridSearchCV(pipe, grid_params,scoring='neg_root_mean_squared_error',
                      cv=cv,n_jobs=10,refit=True,verbose=1,
                      pre_dispatch=20,return_train_score=True)
    gs.fit(X_train,Y_train.values.ravel())
    ### results
    results = pd.DataFrame(gs.cv_results_)
    results = results.sort_values(by='rank_test_score')
    
    return results

#%% compute metrics
def metrics(X_train,Y_train,X_test,Y_test,pipe):
    """
    Compute RMSE and R2 for training and testing sets
    using model in pipe

    Parameters
    ----------
    X_train : pandas DataFrame
        Training set predictors
    Y_train : pandas DataFrame
        Training set target
    X_test : pandas DataFrame 
        Testing set predictors
    Y_test : pandas DataFrame
        Testing set target
    pipe : dict
        pipeline of steps. Last step is model

    Returns
    -------
    RMSE_train : float
        RMSE for training set
    R2_train : float
        R2 for training set
    RMSE_test : float
        RMSE for testing set
    R2_test : float
        R2 for testing set

    """
    ### training set score
    y_pred = pipe.predict(X_train)
    RMSE_train = np.sqrt(mean_squared_error(Y_train, y_pred))
    R2_train = r2_score(Y_train, y_pred)
    ### testing set score
    y_pred = pipe.predict(X_test)
    RMSE_test = np.sqrt(mean_squared_error(Y_test, y_pred))
    R2_test = r2_score(Y_test, y_pred)
    
    return RMSE_train,R2_train,RMSE_test,R2_test


#%%

def calibrate_pollutants(X_train,Y_train,X_test,Y_test,method='MLR',save_results=False):
    """
    Calibrates pollutants.
    There are 4 pollutants that have reference Station measurements: O3, NO2, NO, and PM10

    Parameters
    ----------
    X_train :
        LCS polutant measurements training set
    Y_train : 
        reference station measurements training set
    X_test :
        LCS pollutant measurements testing set
    Y_test :
        reference station measurements testing set
    
    method  : str
        LCS calibration method, default is MLR
    save_results : bool
        save found model, save calibrated data set, save report with model and scoring metrics
    
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
                    
                        

    """
    # calibrate pollutant info
    print('calibrating %s'%pd.DataFrame(Y_train).columns)
    print('using %s as predictors'%[i for i in X_train.columns])
    cv = KFold(n_splits=10,shuffle=True,random_state=92)
    
    # 2 possible calibration methods: MLR and SVR
    if method == 'MLR':
        print('---------Calibrating using MLR---------')
        results_MLR = calibrate_MLR(X_train,Y_train,cv)
        # select best model
        model_n = 0
        alpha = results_MLR.iloc[model_n]['param_model__alpha']
        best_model = Ridge(fit_intercept=True,tol=1e-4,solver='auto',alpha=alpha)
        best_pipe = Pipeline(steps=[('model',best_model)])
        print('Final model chosen')
        print(best_pipe)
        # fit best model
        best_pipe.fit(X_train,Y_train.values.ravel())
        RMSE_train,R2_train,RMSE_test,R2_test = metrics(X_train,Y_train,X_test,Y_test,best_pipe)
        
        print('Scoring results:\n')
        print('Training set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_train,R2_train))
        print('Testing set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_test,R2_test))
        
        ### calibrated pollutant
        pol_name = [i for i in pd.DataFrame(Y_train).columns][0]
        pol_name = pol_name.replace('_Ref','')
        pol_name = pol_name + '_cal_MLR'
        y_cal_MLR = best_pipe.predict(X_test)
        y_cal_MLR[y_cal_MLR<0.]=0.
        y_cal_MLR = pd.DataFrame(y_cal_MLR,columns=[pol_name],index=X_test.index)
        
        y_cal = y_cal_MLR
        gs_results = results_MLR
        
        if save_results:
            print('Saving MLR results')
            fname = pol_name+'_calibration_model_MLR.pkl'
            joblib.dump(best_pipe, fname)
            fname = pol_name+'_MLR_calibrated_dataSet.pkl'
            y_cal_MLR.to_pickle(fname)
            fname = pol_name+'_MLR_calibration_scores.txt'
            f = open(fname,'a')
            print('-------------',file=f)
            print(datetime.now(),file=f)
            print(pol_name+' calibration',file=f)
            print('Algorithm MLR',file=f)
            print('Model: ',best_pipe,file=f)
            print('Scoring results:\n',file=f)
            print('Training set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_train,R2_train),file=f)
            print('Testing set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_test,R2_test),file=f)
            
    
    if method == 'SVR':
        print('---------Calibrating using SVR---------')
        # model search
        results_SVR = calibrate_SVR(X_train,Y_train,cv)
        # model selection
        model_n = 0
        C = results_SVR.iloc[model_n]['param_model__C']
        g = results_SVR.iloc[model_n]['param_model__gamma']
        e = results_SVR.iloc[model_n]['param_model__epsilon']
        best_model = svm.SVR(kernel='rbf',C=C,gamma=g,epsilon=e)
        scaler = StandardScaler()
        best_pipe = Pipeline(steps=[("scaler", scaler), ("model", best_model)]                   )
        print('Best model found')
        print(best_pipe)
        best_pipe.fit(X_train,Y_train.values.ravel())
        RMSE_train,R2_train,RMSE_test,R2_test = metrics(X_train,Y_train,X_test,Y_test,best_pipe)
        print('Scoring results:\n')
        print('Training set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_train,R2_train))
        print('Testing set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_test,R2_test))
    
        # calibrated pollutant 
        y_cal_SVR = best_pipe.predict(X_test)
        y_cal_SVR[y_cal_SVR<0.]=0.
        pol_name = [i for i in pd.DataFrame(Y_train).columns][0]
        pol_name = pol_name.replace('_Ref','')
        pol_name = pol_name + '_cal_SVR'
        y_cal_SVR = pd.DataFrame(y_cal_SVR,columns=[pol_name],index=X_test.index)
        
        y_cal = y_cal_SVR
        gs_results = results_SVR
        
        # save results
        if save_results:
            print('Saving SVR results')
            fname = pol_name+'_calibration_model_SVR.pkl'
            joblib.dump(best_pipe, fname)
            fname = pol_name+'_SVR_calibrated_dataSet.pkl'
            y_cal_SVR.to_pickle(fname)
            fname = pol_name+'_SVR_calibration_scores.txt'
            f = open(fname,'a')
            print('-------------',file=f)
            print(datetime.now(),file=f)
            print(pol_name+' calibration',file=f)
            print('Algorithm SVR',file=f)
            print('Model: ',best_pipe,file=f)
            print('Scoring results:\n',file=f)
            print('Training set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_train,R2_train),file=f)
            print('Testing set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_test,R2_test),file=f)
    
    return y_cal,gs_results,best_pipe,RMSE_train,R2_train,RMSE_test,R2_test
#%% Load calibration models

def load_calibration_models(pol_name):
    """
    Load calibration models found previously via grid search

    Parameters
    ----------
    pol_name : str
        name of pollutant for calibration

    Returns
    -------
    df_cal : padnas DataFrame
        data set with calibrated pollutant appended

    """
    print('---------------------------------\nLoading calibration model for %s\n---------------------------------'%pol_name)
    model_MLR = joblib.load(pol_name+'_cal_MLR_calibration_model_MLR.pkl')
    model_SVR = joblib.load(pol_name+'_cal_SVR_calibration_model_SVR.pkl')
    
    return model_MLR,model_SVR

def plot_gs_results(results):
    """
    plot grid search results for comparison

    Parameters
    ----------
    results : pandas DataFrame
        data set containing scores for training and testing sets
        using grid search

    Returns
    -------
    fig: matplotlib figure
        figure containig plot of grid search results
        for different hyperparameters

    """
    ### plotting variables
    RMSE_cv_mean = -1*results['mean_test_score']
    RMSE_cv_std = results['std_test_score']
    RMSE_tr_mean = -1*results['mean_train_score']
    RMSE_tr_std = results['std_train_score']
    num_models = np.arange(0,results.shape[0])
    num_models = np.arange(0,results.shape[0])
    ### figure
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(111)
    ax1.plot(num_models,RMSE_cv_mean,marker='^',
             color=(0.27,0.51,0.71),markersize=18,markeredgecolor='k',label='C.V. mean RMSE')
    ax1.fill_between(x=num_models,y1=RMSE_cv_mean-RMSE_cv_std,
                     y2=RMSE_cv_mean+RMSE_cv_std,color=(0.53,0.81,0.92))
    ax1.plot(num_models,RMSE_tr_mean,marker='^',
             color='orange',markersize=18,markeredgecolor='k',label='Training mean RMSE')
    ax1.fill_between(x=num_models,y1=RMSE_tr_mean-RMSE_tr_std,
                     y2=RMSE_tr_mean+RMSE_tr_std,color=(0.96,0.64,0.38))

    xrange = num_models
    ymin = min(min(RMSE_cv_mean-RMSE_cv_std),min(RMSE_tr_mean-RMSE_tr_std))
    ymax = max(max(RMSE_cv_mean+RMSE_cv_std),max(RMSE_tr_mean+RMSE_tr_std))
    yrange=np.arange(ymin-0.5,ymax+0.5,0.5)
    ax1.set_xticks(ticks=xrange)
    ax1.set_xticklabels(labels=[str(i) for i in xrange],fontsize=25)
    ax1.set_yticks(ticks=yrange)
    ax1.set_yticklabels(labels=[str(np.round(i,2)) for i in yrange],fontsize=28)
    ax1.set_ylabel('RMSE [$\mu$g/$m^{3}$]',fontsize=30,rotation=90)
    ax1.set_xlabel('Model nº',fontsize=30)
    ax1.legend(loc='lower right',prop={'size':30},ncol=1,framealpha=0.4)
    ax1.grid()
    ax1.set_title('GridSearch CV results',fontsize=35)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(5)
        
    return fig





def calibrate(df,model,X_tot,pollutant,method):
    """
    Calibrate LCS data set using model found previously
    via grid search

    Parameters
    ----------
    df : pandas DataFrame
        data set
    model : sklearn model
            model fitted
    X_tot : pandas DataFrame
        predictors for model
    pollutant : str
            name of pollutant being calibrated
    method : str
            name of method used for calibration

    Returns
    -------
    None.

    """
    y_pred = model.predict(X_tot)
    y_pred[y_pred<0.] = 0.
    y_pred = pd.DataFrame(y_pred,columns=[pollutant+'_cal_'+method],index=X_tot.index)
    
    return
#%% plot calibration results
def cal_plot(ds_ref,ds_cal,save=False):
    """
    Plot time series for claibrated and reference station measurements

    Parameters
    ----------
    ds_ref : pandas Data Frame
        reference station data set
    ds_cal : pandas Data Frame
        calibrated LCS data set
    save : bool, optional
        whether to save or not the plotted figure. The default is False.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.

    """
    
    ##plot
    # The plot is a time series of Reference Sation data set and calibrated LCS data set
    # O3,NO2,NO,PM10
    
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(221)
    
    ax1.plot(ds_ref.index,ds_ref.loc[:,'O3_Ref'],color=(0.27,0.51,0.71),markersize=10,linewidth=1.5,
             markeredgecolor='k',label='O$_3$ Reference Station')
    ax1.plot(ds_cal.index,ds_cal.loc[:,'O3_cal_SVR'],color=(0,0.55,0.55),markersize=10,linewidth=1.5,
             markeredgecolor='k',label='O$_3$ SVR calibration',linestyle='--')
    
    ax1.set_title('RMSE = 7.08\t R$^2$ = 0.90',fontsize=25)
    
       
  
    ymin = 0
    ymax = 85
    yrange=np.arange(ymin,ymax,20)
    
    ax1.set_yticks(ticks=yrange)
    ylabels = [str(np.round(i,2)) for i in yrange]
    ax1.set_yticklabels(labels=ylabels,fontsize=27)
    ax1.set_ylabel('O$_3$ [$\mu$g/$m^{3}$]',fontsize=28,rotation=90)
  
    xrange = ['2021-11-01','2021-11-15','2021-12-01','2021-12-15']
    ax1.set_xticks(ticks=xrange)
    ax1.set_xticklabels(labels=[i for i in xrange],fontsize=20,rotation=45)
    #ax1.set_xlabel('date',fontsize=20)
  
    ax1.tick_params(direction='out', length=4, width=1)
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.legend(loc='upper right',prop={'size':15},ncol=2,framealpha=0.8,edgecolor = 'black',handleheight = 1,handletextpad=0.2)
    ax1.grid(alpha=0.5)
      
  
    ax2 = fig.add_subplot(222)
  
    
    ax2.plot(ds_ref.index,ds_ref.loc[:,'NO2_Ref'],color=(.94,0.5,.5),markersize=10,linewidth=1.5,
             markeredgecolor='k',label='NO$_2$ Reference Station')
    ax2.plot(ds_cal.index,ds_cal.loc[:,'NO2_cal_SVR'],color=(1,.63,.48),markersize=10,linewidth=1.5,
             markeredgecolor='k',label='NO$_2$ SVR calibration',linestyle='--')
    
    ax2.set_title('RMSE = 6.70\t R$^2$ = 0.86',fontsize=25)
    
    
    
    ymin = 0
    ymax = 120
    yrange=np.arange(ymin,ymax,20)
    ax2.set_yticks(ticks=yrange)
    ylabels = [str(np.round(i,2)) for i in yrange]
    ax2.set_yticklabels(labels=ylabels,fontsize=27)
    
    
    
    ax2.set_xticks(ticks=xrange)
    ax2.set_xticklabels(labels=[i for i in xrange],fontsize=20,rotation=45)
    ax2.set_ylabel('NO$_2$ [$\mu$g/$m^{3}$]',fontsize=28,rotation=90)
    #ax2.set_xlabel('date',fontsize=20)
    
    ax2.tick_params(direction='out', length=4, width=1)
    ax2.tick_params(axis='both', which='major', labelsize=22)  
    ax2.legend(loc='upper right',prop={'size':15},ncol=2,framealpha=0.8,edgecolor = 'black',handleheight = 1,handletextpad=0.2)
    ax2.grid(alpha=0.5)
    
    
    ax3 = fig.add_subplot(223)
    
    ax3.plot(ds_ref.index,ds_ref.loc[:,'NO_Ref'],color=(1.,0.65,0),markersize=10,linewidth=1.5,
             markeredgecolor='k',label='NO Reference Station')
    ax3.plot(ds_cal.index,ds_cal.loc[:,'NO_cal_SVR'],color=(0.72,0.53,0.04),markersize=10,linewidth=1.5,
             markeredgecolor='k',label='NO SVR calibration',linestyle='--')
    
    ax3.set_title('RMSE = 5.99\t R$^2$ = 0.88',fontsize=25)
    
    
    ymin = 0
    ymax = 300
    yrange=np.arange(ymin,ymax,50)
    ax3.set_yticks(ticks=yrange)
    ylabels = [str(np.round(i,2)) for i in yrange]
    ax3.set_yticklabels(labels=ylabels,fontsize=27)
    ax3.set_ylabel('NO[$\mu$g/$m^{3}$]',fontsize=28,rotation=90)
    
    
    ax3.set_xticks(ticks=xrange)
    ax3.set_xticklabels(labels=[i for i in xrange],fontsize=20,rotation=45)
    #ax3.set_xlabel('date',fontsize=20)    
    ax3.tick_params(direction='out', length=4, width=1)
    ax3.tick_params(axis='both', which='major', labelsize=22)  
    
    ax3.legend(loc='upper right',prop={'size':15},ncol=2,framealpha=0.8,edgecolor = 'black',handleheight = 1,handletextpad=0.2)
    ax3.grid(alpha=0.5)
  
    ax4 = fig.add_subplot(224)
    
    
    ax4.plot(ds_ref.index,ds_ref.loc[:,'PM10_Ref'],color=(.33,.42,.18),markersize=10,linewidth=1.5,
             markeredgecolor='k',label='PM$_{10}$ Reference Station')
    ax4.plot(ds_cal.index,ds_cal.loc[:,'PM10_cal_SVR'],color=(.4,.8,0.67),markersize=10,linewidth=1.5,
             markeredgecolor='k',label='PM$_{10}$ SVR calibration',linestyle='--')
    ax4.set_title('RMSE = 4.34\t R$^2$ = 0.79',fontsize=25)
    
    
    ymin = 0
    ymax = 160
    yrange=np.arange(ymin,ymax,20)
    ax4.set_yticks(ticks=yrange)
    ylabels = [str(np.round(i,2)) for i in yrange]
    ax4.set_yticklabels(labels=ylabels,fontsize=27)
    ax4.set_ylabel('PM$_{10}$[$\mu$g/$m^{3}$]',fontsize=28,rotation=90)
  
    
    ax4.set_xticks(ticks=xrange)
    ax4.set_xticklabels(labels=[i for i in xrange],fontsize=20,rotation=45)
    #ax4.set_xlabel('date',fontsize=20)
    
    ax4.tick_params(direction='out', length=4, width=1)
    ax4.tick_params(axis='both', which='major', labelsize=22)
    ax4.legend(loc='upper right',prop={'size':15},ncol=2,framealpha=0.8,edgecolor = 'black',handleheight = 1,handletextpad=0.2)
    ax4.grid(alpha=0.5)
  
    #plt.suptitle('Black carbon proxy results\n after testing set denoising\n',fontsize=35)
    
    for axis in ['top','bottom','left','right']:
      ax1.spines[axis].set_linewidth(2)
      ax2.spines[axis].set_linewidth(2)
      ax3.spines[axis].set_linewidth(2)
      ax4.spines[axis].set_linewidth(2)
  
    fig.tight_layout()
    fig.set_size_inches(14,9)
    
    if save:
        plt.savefig("SVR_cal_RefStation_timSeries.png", bbox_inches='tight', dpi=600)
    return fig
    


#%% 
def main():
    """
    Test calibration procedure
    1. Load dataset
    2. pre processing steps
    4. calibration of example pollutant
    
    Returns
    -------
    df_cal: pandas DataFrame
            data set containing calibrated pollutant
            appended to original data set
            
    df_imp: pandas DataFrame
            original data set with imputed values
            pollutants not calibrated
            
    results_MLR: pandas DataFrame 
                MLR calibration gridsearch results
    
    plot_results_MLR: matplotlib figure
                        plot of MLR grid search results 
    
    results_SVR: pandas DataFrame
                SVR calibration gridsearch results
                
    plot_results_SVR: matplotlib figure
                        plot of SVR gridsearch results
                        

    
    
    """
    ############################
    #----- Previous modules ----
    ############################
    # Load data set
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/1_Files/raw_data_files'
    print('Loading data set from %s'%path)
    Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = DS.load_dataSets(path)
    # Pre-processing steps
    print('Pre-processing steps')
    Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = PP.pre_processing(Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N)
    
    ######################
    #---- This module ----
    # testing calibration example
    ######################
    print('-------------------\nCalibration example\n-------------------')
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/test'
    os.chdir(path)
    pol = 'PM10'
    print('Calibrating %s'%pol)
    
    # generate data sets
    df_cal = calibration_datasets(Ref_O3,Ref_NO2,Ref_NO,Ref_PM10,Ref_Meteo,LCS_O3,LCS_NO2,LCS_NO,LCS_PM10,LCS_Meteo,cal_pol=pol)
    # data set split
    print('Train/test split')
    X_train,Y_train,X_test,Y_test = calibration_train_test_split(df_cal,test_frac=0.25)
    
    """    
    # calibration example
    Ref = Ref_O3.copy()
    Ref.dropna(inplace=True)
    Ref.index = Ref.date
    Ref = Ref.O3_Ref
    Ref = pd.DataFrame(Ref)
    Ref_tr = Ref.loc[Ref.index.isin(X_train.index)]
    Ref_tr = Ref_tr[~Ref_tr.index.duplicated(keep='first')]
    Ref_test = Ref.loc[Ref.index.isin(X_test.index)]
    Ref_test = Ref_test[~Ref_test.index.duplicated(keep='first')]
    
    X_cal_training = X_train.loc[:,['S_O3','T_int','RH_int']]
    ds_tr = pd.concat([Ref_tr,X_cal_training],axis=1)
    ds_tr.dropna(inplace=True)
    X_cal_test = X_test.loc[:,['S_O3','T_int','RH_int']]
    ds_test = pd.concat([Ref_test,X_cal_test],axis=1)
    ds_test.dropna(inplace=True)
    """
    print('-------------------------------\nThe calibration process returns\n1. calibrated data set\n2. grid search results\n3. calibration model\n4. scoring metrics RMSE/R2 for training and testing set\n')
    y_cal,gs_results,best_pipe,RMSE_train,R2_train,RMSE_test,R2_test = calibrate_pollutants(X_train,Y_train,X_test,Y_test,method='MLR',save_results=True)

    
    return y_cal,gs_results,best_pipe,RMSE_train,R2_train,RMSE_test,R2_test

    
   
if __name__ == '__main__':
    y_cal,gs_results,best_pipe,RMSE_train,R2_train,RMSE_test,R2_test =  main()