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

### ML
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn import svm
from mlxtend.feature_selection import SequentialFeatureSelector as SFS 
### modules
import BCP_DataSet as DS
import BCP_PreProcessing as PP
#%%
"""
Functions

"""
def split(df,test_frac=0.25,n_splits=10):
    """
    split data set into train/test sets
    
    df: dataFrame containng measurements
    test_frac: fraction of samples for testing set
    """
    ### BC is first column of data frame
    X_tot = df.iloc[:,1:]
    Y_tot = df.iloc[:,0]
    X_train, X_test, Y_train, Y_test = train_test_split(X_tot,Y_tot,test_size=test_frac,shuffle=True,random_state=92)
    cv = KFold(n_splits=n_splits,shuffle=True,random_state=92)
    print("Dataset splitted\nTrain set size %s \tTest set size %s" %(X_train.shape[0],X_test.shape[0]))
    print('%i predictors:\n'%X_train.shape[1])
    print([i for i in X_train.columns])
    return X_train, X_test, Y_train, Y_test,cv

def SVR_BFS(cv,X_train,Y_train,max_features,min_features,path ='/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/3_BC_Proxy/Feature_Selection/BFS',test_run=False):
    """
    SVR BFS
    
    cv: cross-validation split
    X_train/Y_train: training set for BC and its predictors
    max_features: maximum number of predictors for feature selection
    min_features: minimum number of predictors for feature selection
    path: directory for saving results
    """    
    ### change directory
    os.chdir(path)
    ### gridsearch parameters
    grid_params = {
        'bfs__estimator__C':np.logspace(-5,15,11,base=2),
        'bfs__estimator__gamma':np.logspace(-15,3,10,base=2),
        'bfs__estimator__epsilon':np.logspace(-6,0,5,base=2)
    }
    
    start_time = time.time()
    first = True
    ### iterate over the grid
    for C in grid_params.get('bfs__estimator__C'):
        for gamma in grid_params.get('bfs__estimator__gamma'):
            for epsilon in grid_params.get('bfs__estimator__epsilon'):
                ### pipeline
                model = svm.SVR(kernel='rbf',C=C,gamma=gamma,epsilon=epsilon)
                scaler = StandardScaler()

                print('Starting BFS for model')
                print(model)
        
                bfs = SFS(model,k_features=(min_features,max_features),forward=False,verbose=2,
                              scoring='neg_root_mean_squared_error',cv=cv,n_jobs=10,
                              pre_dispatch=10,clone_estimator=True)
                bfs.fit(scaler.fit_transform(X_train),Y_train)
                if first:
                    ### Create DataFrame to store results
                    df = pd.DataFrame(bfs.get_metric_dict())
                    ### add hyperparameters to df
                    df.loc['model_C'] = C
                    df.loc['model_gamma'] = gamma
                    df.loc['model_epsilon'] = epsilon
                
                    ### save DataFrame
                    if not test_run:
                        fname = 'Results_BC_Proxy_BFS_SVR_provisional.pkl'
                        df.to_pickle(fname)
                
                    first = False
                
                else:
                    df1 = pd.DataFrame(bfs.get_metric_dict())   
                    ### add model's hyperparameters
                    df1.loc['model_C'] = C
                    df1.loc['model_gamma'] = gamma
                    df1.loc['model_epsilon'] = epsilon
                
                    ### concat
                    df = pd.concat([df,df1],axis=1)
                    ### save DataFrame
                    if not test_run:
                        fname = 'Results_BC_Proxy_BFS_SVR_provisional.pkl'
                        df.to_pickle(fname)
                
                
                print('Model finished\n')
                

    ### save final results
    if not test_run:
        fname = 'Results_BC_Proxy_BFS_SVR_max'+str(max_features)+'_min'+str(min_features)+'.pkl'
        df.to_pickle(fname)
    else:
        print('Testing.\n No file will be saved to disk')
        
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
def main():
    ### load dataset
    BC, LCS_PM1, LCS_PM25, LCS_N, Meteo, O3_cal, NO2_cal, NO_cal, PM10_cal = DS.Pollutants()
    ### Pre-processing steps
    dataSet = PP.pre_processing(BC, LCS_PM1, LCS_PM25, LCS_N, Meteo, O3_cal, NO2_cal, NO_cal, PM10_cal)
    ### Feature Selection
    print('Starting SVR-BFS')
    X_train, X_test, Y_train, Y_test,cv = split(dataSet,test_frac=0.25)
    BFS_results = SVR_BFS(cv,X_train,Y_train,max_features=X_train.shape[1],min_features=1,test_run=True)
    Explore_results(BFS_results, X_train, Y_train, cv=cv, max_features=X_train.shape[1])

    return BFS_results
    

if __name__ == '__main__':
    BFS_results = main()