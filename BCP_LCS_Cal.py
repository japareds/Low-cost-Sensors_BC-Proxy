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
### ML
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn import svm
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

### plot
import matplotlib.pyplot as plt

### Modules
import BCP_DataSet as DS
import BCP_PreProcessing as PP
import BCP_Imputation as IMP

#%% functions

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
        'model__alpha': np.linspace(0,10,30),
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
        'model__C': [46,100,10],#np.logspace(0,3,5,base=10),
        'model__gamma': [0.825,3.0,0.1,10.0],#np.logspace(-2,1,5,base=10),
        'model__epsilon': [0.5,0.4],#np.logspace(-2,0,5,base=10)
        }

    gs = GridSearchCV(pipe, grid_params,scoring='neg_root_mean_squared_error',
                      cv=cv,n_jobs=10,refit=True,verbose=1,
                      pre_dispatch=20,return_train_score=True)
    gs.fit(X_train,Y_train.values.ravel())
    ### results
    results = pd.DataFrame(gs.cv_results_)
    results = results.sort_values(by='rank_test_score')
    
    return results
    
### plot grid search results
def gs_results(results):
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

### compute metrics
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



#%% 
def main():
    """
    Test calibration procedure
    1. Load dataset
    2. pre processing steps
    3. imputation
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
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/1_Files/raw_data_files'

    ### load dataset
    Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = DS.load_or_create(path,load=True)
    ### Pre-processing steps
    df = PP.pre_processing(Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N)
    ### simple imputation
    method ='remove'
    df_imp, imp = IMP.imputation(df, method=method,is_train=True)
    ### calibration example
    print('----Calibration example----')
    print('calibrating NO2')
    Y_tot = df_imp[['NO2_Ref']]
    X_tot = df_imp[['S_NO2','S_O3','T_int','RH_int']]
    X_train, X_test, Y_train, Y_test = train_test_split(X_tot,Y_tot,test_size=0.25,shuffle=True,random_state=92)
    cv = KFold(n_splits=10,shuffle=True,random_state=92)
    #### MLR
    print('---------Calibrating using MLR---------')
    results_MLR = calibrate_MLR(X_train,Y_train,cv)
    plot_results_MLR = gs_results(results_MLR)
    model_n = 0
    alpha = results_MLR.iloc[model_n]['param_model__alpha']
    final_model = Ridge(fit_intercept=True,tol=1e-4,solver='auto',alpha=alpha)
    final_pipe = Pipeline(steps=[('model',final_model)])
    print('Final model chosen')
    print(final_pipe)
    final_pipe.fit(X_train,Y_train.values.ravel())
    RMSE_train,R2_train,RMSE_test,R2_test = metrics(X_train,Y_train,X_test,Y_test,final_pipe)
    print('Scoring results:\n')
    print('Training set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_train,R2_train))
    print('Testing set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_test,R2_test))
    ### calibrated pollutant
    y_cal_MLR = final_pipe.predict(X_tot)
    y_cal_MLR[y_cal_MLR<0.]=0.
    y_cal_MLR = pd.DataFrame(y_cal_MLR,columns=['NO2_cal_MLR'],index=X_tot.index)
    
    #### SVR
    print('---------Calibrating using SVR---------')
    results_SVR = calibrate_SVR(X_train,Y_train,cv)
    plot_results_SVR = gs_results(results_SVR)
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
    RMSE_train,R2_train,RMSE_test,R2_test = metrics(X_train,Y_train,X_test,Y_test,final_pipe)
    print('Scoring results:\n')
    print('Training set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_train,R2_train))
    print('Testing set\n RMSE = %.2f \t R2 = %.2f'%(RMSE_test,R2_test))
    #### calibrated pollutant 
    y_cal_SVR = final_pipe.predict(X_tot)
    y_cal_SVR[y_cal_SVR<0.]=0.
    y_cal_SVR = pd.DataFrame(y_cal_SVR,columns=['NO2_cal_SVR'],index=X_tot.index)
    ### add calibrated pollutant to data set
    df_cal = pd.concat([df_imp,y_cal_MLR,y_cal_SVR],axis=1)
    
    return df_cal,df_imp,results_MLR,plot_results_MLR,results_SVR,plot_results_SVR

    
   
if __name__ == '__main__':
    ds_cal,ds,results_MLR,plot_results_MLR,results_SVR,plot_results_SVR =  main()