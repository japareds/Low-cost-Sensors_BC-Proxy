#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBJECTIVE:
    Data imputation of missing values
    
Created on Tue Jun 28 13:11:07 2022

@author: jparedes
"""
### data manipulation and computation
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
### ML
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LR


### modules
import BCP_DataSet as DS
import BCP_PreProcessing as PP


#%%
"""
   The proxy requires that all predictors 
   have simultaneous measurements
   
Functions:
    different imputation methods
    - MICE
    - MissForest
    - KNN

"""
def nan_counter(df):
    """
    Report how many points are lost
    
    Parameters
    ----------
    df: pandas DataFrame - DataFrame with nan entries
    
    Returns
    -------
    missing_counter: float - percentage of missing values 
    for all columns
    """
    
    n_total = df.shape[0]
    missing_counter = 100*df.isna().sum()/n_total    
    return missing_counter

    
def remove_nan(df):
    """
    Simply remove nan values
    
    Parameters
    ----------
    df: pandas DataFrame
        DataFrame containing NaNs
    
    Returns
    -------
    df: pandas DataFrame
        DataFrame without NaN entries
    """
    
    ### missing values
    df.dropna(inplace=True)
    return df,None

def mice(df, n_iter, is_train,imp=None):
    """
    Multiple imputation using sklearn IterativeImputer
    each feature is estimated from all the others
    
    - sample_posterior set to False because RFR doesnt return std
    - n_neares_features set to None to use all features. The selection is done before entering the function
    

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with NaN entries
        
    n_iter: int
        number of imputation rounds iterations
        
    is_train: bool
        asking if the imputed variable data set belongs to
        the training set or not(testing set)
        
    imp: sklearn imputer
        imputer fitter from training phase
        in case to apply it for testing set

    Returns
    -------
    df_: pandas DataFrame
        DataFrame with NaN entries imputed
        
    imp: sklearn imputer
        imputer fitted
    """
    
    ### copy data frame for imputation
    df_ = df.copy()
    df_[df_.isna()] = np.nan

    ### model: Linear Regression 
    estimator = LR(fit_intercept=True, copy_X=True,
                   n_jobs=2, positive=False)
    ### fit the imputer only for the training set 
    ### for the testing set use the imputer already fitted
    if is_train:
    
        imp = IterativeImputer(estimator=estimator, 
                               missing_values=np.nan,
                               sample_posterior=False, 
                               max_iter=n_iter, 
                               n_nearest_features=None, 
                               initial_strategy='mean',imputation_order='ascending',
                               min_value=0.0, max_value=np.inf,
                               verbose=0, random_state=92,
                               skip_complete=False, add_indicator=False,tol=0.01,)
        df_ = imp.fit_transform(df_)
    
    else:
        df_= imp.transform(df_)
    
    ### convert to pandas DataFrame
    df_ = pd.DataFrame(df_)
    df_.columns = df.columns
    df_.index = df.index
    
    return df_,imp


def miss_forest(df,n_iter, is_train,imp=None):
    """
    Multiple imputation using sklearn IterativeImputer
    each feature is estimated from all the others
    
    - sample_posterior set to False because RFR doesnt return std
    - n_neares_features set to None to use all features. The selection is done before entering the function
    
    Parameters
    ----------
    df: pandas DataFrame
        DataFrame containing nan entries
        
    n_iter: int
        number of iterations for multiple imputation
        
    is_train: bool
        asking if the imputed variable data set belongs to
        the training set or not(testing set)
        
    imp: sklearn imputer
        imputer fitted from training phase
        in case to apply it to testing set

        
    Returns
    -------
    df_: pandas DataFrame
        DataFrame with imputed values
        
    imp: sklearn imputer
        fitted imputer
    """
    ### copy data frame for imputation
    df_ = df.copy()
    # format
    df_[df_.isna()] = np.nan

    estimator = RFR(criterion='squared_error', 
                    n_estimators = 1000,max_depth = None,
                    min_samples_split=2,max_features='auto',
                    min_samples_leaf=2,max_leaf_nodes=None,
                    bootstrap=True, oob_score=False, 
                    random_state=92,n_jobs=2,
                    verbose=1, warm_start=False)
    
    ### fit the imputer only for the training set 
    if is_train:
        imp = IterativeImputer(estimator=estimator, 
                               missing_values=np.nan,
                               sample_posterior=False, 
                               max_iter=n_iter, 
                               n_nearest_features=None, 
                               initial_strategy='mean',imputation_order='ascending',
                               min_value=0.0, max_value=np.inf,
                               verbose=1, random_state=92,
                               skip_complete=False, add_indicator=False,tol=0.01)
        
        df_ = imp.fit_transform(df_)
        
        
    else:
        df_= imp.transform(df_)

    ### convert to pandas DataFrame
    df_ = pd.DataFrame(df_)
    df_.columns = df.columns
    df_.index = df.index
    
    return df_,imp

def knn(df,k,is_train,pipe=None):
    """
    KNN imputation method
    
    Parameters
    ----------
    df: pandas DataFrame
        DataFrame with nan entries
        
    k: int
        number of neighbours for knn algorithm

    is_train: bool
        asking if the imputed variable data set belongs to
        the training set or not(testing set)
        
    pipe: sklearn pipeline with imputer
        fitted imputer from trainnig phase
        in case to apply it in the testing set
    
    
    Returns
    -------
    df_: pandas DataFrame 
        DataFrame with imputed values
    
    """
    
    ### copy data frame for imputation
    df_ = df.copy()
    df_[df_.isna()] = np.nan
    ### scale distances
    scaler = MinMaxScaler()
    ### imputation
    imputer = KNNImputer(missing_values=np.nan,n_neighbors=k,
                         weights='uniform',metric='nan_euclidean',
                         copy=True,add_indicator=False)
    ### fit the imputer only for the training set 
    if is_train:    
        pipe = Pipeline(steps=[('scaler',scaler),('knn_imputer',imputer)])
        ### pipeline
        df_= pipe.fit_transform(df_)
    else:
        df_= pipe.transform(df_)

    ### reverse scaler transform
    df_ = pipe.named_steps['scaler'].inverse_transform(df_)
    ### convert to pandas DataFrame
    df_ = pd.DataFrame(df_)
    df_.columns = df.columns
    df_.index = df.index
    return df_,pipe


def variables(df,imp_pollutant,ref=True):
    """
    Multiple imputation estimates the value based on predictors
    This function checks wether the corresponding variable 
    is together the required predictors

    Parameters
    ----------
    df: pandas DataFrame
        data frame containing feature of interest in column 0 
        and the predictors in the next columns
    
    imp_pollutant: str
        name of the pollutant to be imputed
        
    ref = bool, optional
        wether the pollutant was measured by 
        a reference station or a LCS, default is True
        

    Returns
    -------
    df_: pandas DataFrame
        data set to be imputed
        df_ is a subset of the original dataset df
        the first column correponds to the variable
        to be imputed and the others are the predictors
        
    """
    ### imputation of variables
    ### The first column corresponds to 
    ### the variable to be imputed
    
    ### reference station measurements
    if ref:
        print('Imputation of air pollutants measured by Reference Station')
        if imp_pollutant == 'O3':
            var = ['O3_Ref','T','RH']
        if imp_pollutant == 'NO2':
            var = ['NO2_Ref','T','RH']
        if imp_pollutant == 'NO':
            var = ['NO_Ref','T','RH']
        if imp_pollutant == 'PM10':
            var = ['PM10_Ref','T','RH']
        if imp_pollutant == 'N_Ref':
            var = ['N_Ref','T','RH']
        if imp_pollutant == 'T':
            var = ['T','RH']
        if imp_pollutant == 'RH':
            var = ['RH','T']
    
    ### LCS measurements
    else:
        if imp_pollutant == 'O3':
            var = ['S_O3','T_int','RH_int']
        if imp_pollutant == 'NO2':
            var = ['S_NO2','T_int','RH_int']
        if imp_pollutant == 'NO':
            var = ['S_NO','T_int','RH_int']
        if imp_pollutant == 'T':
            var = ['T_int','RH_int']
        if imp_pollutant == 'RH':
            var = ['RH_int','T_int']
        if imp_pollutant == 'PM1':
            var = ['S5_PM1','T','RH']
        if imp_pollutant == 'PM25':
            var = ['S5_PM25','T','RH']
        if imp_pollutant == 'PM10':
            var = ['S5_PM10','T','RH']
        if imp_pollutant == 'N0.3':
            var = ['S5_0.3','T','RH']
        if imp_pollutant == 'N0.5':
            var = ['S5_0.5','T','RH']
        if imp_pollutant == 'N1.0':
            var = ['S5_1.0','T','RH']
        if imp_pollutant == 'N2.5':
            var = ['S5_2.5','T','RH']
        if imp_pollutant == 'N5.0':
            var = ['S5_5.0','T','RH']
        if imp_pollutant == 'N10.0':
            var = ['S5_10.0','T','RH']
    ### create the data frame to be imputed
    df_ = df[var]
    return df_


def imputation(df, method,is_train=True,imp=None,n_iter=10,k=5):
    """
    Missing values imputations
    

    Parameters
    ----------
    df : pandas DataFrame
        Data frame containing the missing values as NaN
        column 0 must be the variable for imputation and the others are the predictors.
        
    method: str
        Imputation method to use 
        
    is_train: bool
        asking if the imputed variable data set belongs to
        the training set or not(testing set)

    imp: sklearn imputer
        previous imputer fitted from training data
        
    n_iter: int
        number of iterations for multiple imputation
        
    k: int 
        Parameter for KNN imputation
    
    Returns
    -------
    new_df: pandas DataFrame
        Data Frame with imputed values
    
    imp: sklearn imputer
        imputer object fitted with training data

    """
    print('Data set contains the following entries: ',[i for i in df.columns])
    ### count percentage of missing values (nan)
    missing_percentage = nan_counter(df)
    print('Percentage of missing values:\n')
    print(missing_percentage) 
    
    ### imputation method
    variables = [i for i in df.columns]
    print('Imputation of %s using %s as predictors'%(variables[0],variables[1:]))
    print('Imputation method: %s'%method)
    if method == 'remove':
        new_df,imp = remove_nan(df)
    elif method == 'MICE':
        new_df,imp = mice(df,n_iter,is_train,imp=imp)
    elif method == 'MissForest':
        new_df,imp = miss_forest(df,n_iter,is_train,imp=imp)
    elif method == 'KNN':
        new_df,imp = knn(df,k,is_train,pipe=imp)
    
    ### the relevant imputed feature is column 0 of data frame
    #imputed_feature = new_df.iloc[:,0]
    
    return new_df,imp

def impute_target(Y,is_train=True,value=None):
    """
    Imputation of target variable (BC)
    The median is used

    Parameters
    ----------
    Y : pandas Series
        BC concentration for different times
    
    is_train: bool
            boolean specifying if Y belongs to the training set
    
    value = float
            number value to use to fill missing values
    
    Returns
    -------
    imp_Y : pandas Series
        BC concentration with imputed values

    """
    miss = 100*Y.isna().sum()/Y.shape[0]
    print('BC missing percentage %.2f '%miss)
    if is_train:
        value = Y.median()       
        
    imp_Y = Y.fillna(value=value)
    
    return imp_Y,value



#%%
"""
main
"""

def main():
    """
    Test imputation methods  

    Returns
    -------
    df: pandas DataFrame. 
        Contains original dataFrame with missing values (nan)
    
    new_df: pandas DataFrame.
            New dataFrame with imputed values
    """
    
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/1_Files/raw_data_files'

    ### load dataset
    Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = DS.load_or_create(path,load=True)
    ### Pre-processing steps
    df = PP.pre_processing(Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N)
    ### test imputation fitting the imputer and then using it to transform the testing set
    Y_tot = df.iloc[:,0]
    X_tot = df.iloc[:,1:]
    X_train, X_test, Y_train, Y_test = train_test_split(X_tot,Y_tot,test_size=0.25,shuffle=True,random_state=92)
    print('---------------\nTesting imputation techniques\n---------------')
    print('Creating imputer from training set')
    
    ### pollutants to impute
    pollutants = ['O3','NO2','NO','PM10','N_Ref','T','RH','O3','NO2','NO','T','RH','PM1','PM25','PM10','N0.3','N0.5','N1.0','N2.5','N5.0','N10.0']
    ref = [True,True,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False,False,False,False,False]
    imp_pollutants = []
    imputers = []
    
    ### imputation
    method = 'MICE'
    for p,r in zip(pollutants,ref):
        df = variables(X_train,imp_pollutant=p,ref=r)
        imp_pol,imp = imputation(df, method=method,is_train=True)
        
        ### save pollutant with imputed values
        pollutant_imp = pd.DataFrame(imp_pol.iloc[:,0])
        imp_pollutants.append(pollutant_imp)
        imputers.append(imp)
        
    ### joint imputed pollutants into one data frame
    new_X_train = pd.DataFrame()
    for i in range(len(imp_pollutants)):
        new_X_train[imp_pollutants[i].columns[0]] = imp_pollutants[i].iloc[:,0]

    ### apply the imputers to the testing set
    print('Transforming testing set')
    imp_pollutants = []
    idx = [i for i in range(0,len(ref))]
    
    ### imputation
    for p,r,i in zip(pollutants,ref,idx):
        df = variables(X_test,imp_pollutant=p,ref=r)
        
        imp_pol,imp = imputation(df, method=method,is_train=False,imp=imputers[i])
        
        ### save pollutant with imputed values
        pollutant_imp = pd.DataFrame(imp_pol.iloc[:,0])
        imp_pollutants.append(pollutant_imp)
        imputers.append(imp)
        
    ### updated X_test with imputed values
    new_X_test = pd.DataFrame()
    for i in range(len(imp_pollutants)):
        new_X_test[imp_pollutants[i].columns[0]] = imp_pollutants[i].iloc[:,0]
        
    ### imputation of BC
    imp_Y_train,BC_median =  impute_target(Y_train,is_train=True,value=None)
    imp_Y_test,BC_median =  impute_target(Y_test,is_train=False,value=BC_median)
    

    print('----------\nTesting Finished\n----------\n')
    
    return df,X_train,X_test,new_X_train,new_X_test,imp,Y_train,Y_test,imp_Y_train,imp_Y_test

if __name__ == '__main__':
    ds,X_train,X_test,new_X_train,new_X_test,imp,Y_train,Y_test,imp_Y_train,imp_Y_test = main()

