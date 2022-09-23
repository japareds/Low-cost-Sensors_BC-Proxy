#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBJECTIVE:
    Feature Selection
    
    Algorithm: SVR
    Wrapper method: Backwards feature selection (BFS)
    Metric: RMSE

Created on Mon Jun 27 13:01:44 2022

@author: jparedes
"""



### utilities
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### ML
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import  KFold, cross_validate, learning_curve,validation_curve,GridSearchCV
from sklearn import svm
from mlxtend.feature_selection import SequentialFeatureSelector as SFS 
### modules
import BCP_DataSet as DS
import BCP_PreProcessing as PP
#%%
"""
Functions

"""

def SVR_BFS(X_train,Y_train,max_features,min_features,save_results=False):
    """
    Backwards Feature Selection wrapper method
    SVR ML algorithm
    
    Parameters
    ----------
    
    X_train : pandas DataFrame
            proxy predictors training set
    Y_train: pandas DataFrame
            BC measurements training set
    max_features: int
            maximum number of predictors for starting feature selection
    min_features: int
            minimum number of predictors for stopping feature selection
    save_results : bool
            save BFS results to directory
    """    
    #######################################
    # The BFS built in implementation only considers 1 model
    # Thus, it is necessary to optimize hyperparameters
    #######################################
    
    # gridsearch SVR hyperparameters
    grid_params = {
        'bfs__estimator__C':np.logspace(-5,15,11,base=2),
        'bfs__estimator__gamma':np.logspace(-15,3,10,base=2),
        'bfs__estimator__epsilon':np.logspace(-6,0,5,base=2)
    }
    cv = KFold(n_splits=10,shuffle=True,random_state=92)
    
    start_time = time.time()
    first = True
    ### iterate over the grid
    for C in grid_params.get('bfs__estimator__C'):
        for gamma in grid_params.get('bfs__estimator__gamma'):
            for epsilon in grid_params.get('bfs__estimator__epsilon'):
                
                model = svm.SVR(kernel='rbf',C=C,gamma=gamma,epsilon=epsilon)
                scaler = StandardScaler()

                print('\n\n----------------------\nStarting BFS for model\n----------------------\n\n')
                print(model)
        
                # define BFS and fit
                bfs = SFS(model,k_features=(min_features,max_features),forward=False,verbose=2,
                              scoring='neg_root_mean_squared_error',cv=cv,n_jobs=10,
                              pre_dispatch=10,clone_estimator=True)
                bfs.fit(scaler.fit_transform(X_train),Y_train)
                
                # store BFS results
                if first:
                    # first time, create data frame
                    df = pd.DataFrame(bfs.get_metric_dict())
                    # add hyperparameters to df
                    df.loc['model_C'] = C
                    df.loc['model_gamma'] = gamma
                    df.loc['model_epsilon'] = epsilon
                
                    # save DataFrame
                    if save_results:
                        fname = 'Results_BC_Proxy_BFS_SVR_provisional.pkl'
                        df.to_pickle(fname)
                
                    first = False
                
                else:
                    df1 = pd.DataFrame(bfs.get_metric_dict())   
                    # add model's hyperparameters
                    df1.loc['model_C'] = C
                    df1.loc['model_gamma'] = gamma
                    df1.loc['model_epsilon'] = epsilon
                
                    # add new results to previous results
                    df = pd.concat([df,df1],axis=1)
                    # save updated results
                    if save_results:
                        fname = 'Results_BC_Proxy_BFS_SVR_provisional.pkl'
                        df.to_pickle(fname)
                
                
                print('\nModel finished\n')
                

    # save final results
    if save_results:
        fname = 'Results_BC_Proxy_BFS_SVR_max'+str(max_features)+'_min'+str(min_features)+'.pkl'
        df.to_pickle(fname)
    else:
        print('No file will be saved to disk')
        
    #Notification()
    end_time = time.time()
    print('BFS Finished in %.2f'%(end_time-start_time))
    

    
    return df

def Explore_results(df,X_train,Y_train,cv,max_features,path ='/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/3_BC_Proxy/Feature_Selection/BFS',fname="Report_SVR_BFS.txt",test_run=False):
    """
    Explore the results for BFS
    For each feature, select the best model
   
   
    Parameters
    ----------
    df : dataFrame with results from BFS
    path : directory where BFS results are located
    max_features : maximum number of features considered
    fname: file name of the report

    Returns
    -------
    None.

    A txt file is created at path
    """
    ### create file to store best results for every feature set size
    os.chdir(path)
    f = open(fname, "w")
    print('SVR Model BFS',file=f)
    RMSE_mean = []
    RMSE_std = []
    RMSE_mean_training = []
    RMSE_std_training = []
    RMSE_ci = []
    num_features = []
    for i in range(max_features,0,-1):
        print('\n Considering a set of ',i,'features',file=f)
        ### best scoring for that feature set size
        results_feature_size = df[[i]]
        errors = np.array(-1.*results_feature_size.loc['avg_score'])
        std = np.array(results_feature_size.loc['std_dev'])
        confidence_intervals = np.array(results_feature_size.loc['ci_bound'])
        idx = np.argmin(errors)
        predictors = np.array(results_feature_size.loc['feature_names'])
        model_C = np.array(results_feature_size.loc['model_C'])
        model_gamma = np.array(results_feature_size.loc['model_gamma'])
        model_epsilon = np.array(results_feature_size.loc['model_epsilon'])
    
    
        print('minimum error obtained: ',min(errors),'+/-',std[idx])
        print('Model Hyperparameters: \n C = %.1f \n gamma = %.1f \n epsilon = %.1f'
              %(model_C[idx],model_gamma[idx],model_epsilon[idx]))
        print('Predictors set:\n ',predictors[idx])
        ### save the results to file
        print('minimum error obtained: ',min(errors),'+/-',std[idx],file=f)
        print('Model Hyperparameters: \n C = %.1f \n gamma = %.1f \n epsilon = %.1f'
              %(model_C[idx],model_gamma[idx],model_epsilon[idx]),file=f)
        print('Predictors set:\n ',predictors[idx],file=f)
        print('Cross validation results:\n',file=f)
        model = svm.SVR(C=model_C[idx],gamma=model_gamma[idx],epsilon=model_epsilon[idx])
        scaler = StandardScaler()
        pipe = Pipeline(steps=[('scaler',scaler),('model',model)])
        scores = cross_validate(pipe, X_train, Y_train,
                                scoring='neg_root_mean_squared_error', cv=cv,
                                n_jobs=10, verbose=1,
                                return_train_score=True)
        print('CV scores: %.2f +/- %.2f\nCV training scores: %.2f +/- %.2f'
              %(-1*scores['test_score'].mean(),scores['test_score'].std(),
                -1*scores['train_score'].mean(),scores['train_score'].std()),
              file=f)
        ### save the results for plotting
        RMSE_mean.append(min(errors))
        RMSE_std.append(std[idx])
        RMSE_mean_training.append(-1*scores['train_score'].mean())
        RMSE_std_training.append(scores['train_score'].std())
        RMSE_ci.append(confidence_intervals[idx])
        num_features.append(i)
    
    

    f.close()

    return 
#%%
def model_curves(pipe,X_train,Y_train):
    grid_params = {
        'model__C':np.logspace(-3,3,7),
        'model__gamma':np.logspace(-3,3,7),
        'model__epsilon':np.linspace(0,0.8,4)
        }
    cv = KFold(n_splits=10,shuffle=True,random_state=92)
    
    val_curve = True
    if val_curve:
        C = 1
        gamma = 1
        epsilon = 0.2
        model = svm.SVR(kernel='rbf',C=C,gamma=gamma,epsilon=epsilon)
        scaler = StandardScaler()
        model_BC = Pipeline(steps=[('scaler',scaler),('model',model)])
        param_range = np.logspace(-3, 3,7)
        train_scores, test_scores = validation_curve(model_BC, X_train, Y_train, param_name='model__C', param_range=param_range, groups=None, cv=cv, scoring='neg_root_mean_squared_error', 
                                                     n_jobs=10, pre_dispatch='all', verbose=2, error_score=np.nan, fit_params=None)
        
        train_scores_mean = -np.mean(train_scores, axis=1)
        train_scores_std = -np.std(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        test_scores_std = -np.std(test_scores, axis=1)
        
        fig = plt.figure(figsize=(20,10))
        ax1 = fig.add_subplot(111)
        ax1.semilogx(param_range,train_scores_mean,marker='^',color=(0.27,0.51,0.71),markersize=10,
                 markeredgecolor='k',label='Training scores')
        ax1.fill_between(x=param_range,y1=train_scores_mean-train_scores_std,
                         y2=train_scores_mean+train_scores_std,color=(0.53,0.81,0.92))
        
        ax1.semilogx(param_range,test_scores_mean,marker='^',color=(0,0.55,0.55),markersize=10,
                 markeredgecolor='k',label='Test scores')
        ax1.fill_between(x=param_range,y1=test_scores_mean-test_scores_std,
                         y2=test_scores_mean+test_scores_std,color=(0.37,0.62,0.63))
              
      
        ymin = 0.0
        ymax = 1.2
        yrange=np.arange(ymin,ymax,0.2)
        
        ax1.set_yticks(ticks=yrange)
        ylabels = [str(np.round(i,2)) for i in yrange]
        ax1.set_yticklabels(labels=ylabels,fontsize=20)
        ax1.set_ylabel('RMSE [$\mu$g/$m^{3}$]',fontsize=20,rotation=90)
      
        xrange = param_range
        ax1.set_xticks(ticks=xrange)
        ax1.set_xticklabels(labels=[str(i) for i in xrange],fontsize=20)
        ax1.set_xlabel('C',fontsize=20)
      
        ax1.legend(loc='upper right',prop={'size':15},ncol=1,framealpha=0.4)
        ax1.grid(alpha=0.5)
        
        plt.suptitle('SVR validation curve $\gamma$ = 1')
    
    
    return fig

def hyperparameters_search(X_train,Y_train):
    grid_params = {
        'model__C':np.logspace(-1,1,3),
        'model__gamma':np.logspace(-1,1,3),
        'model__epsilon':np.linspace(0,1,9)
        }
    cv = KFold(n_splits=10,shuffle=True,random_state=92)
    
    model = svm.SVR(kernel='rbf')
    scaler = StandardScaler()
    model_BC = Pipeline(steps=[('scaler',scaler),('model',model)])
    
    grid = GridSearchCV(model_BC, param_grid = grid_params,
                        scoring='neg_root_mean_squared_error', n_jobs=10, refit=True, cv=cv,
                        verbose=2, pre_dispatch='2*n_jobs', 
                        error_score=np.nan, return_train_score=True)
    
    grid.fit(X_train,Y_train)
    return grid
                    
    
    
#%%
def main():
    ### load dataset
    BC, LCS_PM1, LCS_PM25, LCS_N, Meteo, O3_cal, NO2_cal, NO_cal, PM10_cal = DS.Pollutants()
    ### Pre-processing steps
    dataSet = PP.pre_processing(BC, LCS_PM1, LCS_PM25, LCS_N, Meteo, O3_cal, NO2_cal, NO_cal, PM10_cal)
    ### Feature Selection
    print('Starting SVR-BFS')
    X_train, X_test, Y_train, Y_test,cv = train_test_split(dataSet,test_frac=0.25)
    BFS_results = SVR_BFS(cv,X_train,Y_train,max_features=X_train.shape[1],min_features=1,test_run=True)
    Explore_results(BFS_results, X_train, Y_train, cv=cv, max_features=X_train.shape[1])

    return BFS_results
    

if __name__ == '__main__':
    BFS_results = main()