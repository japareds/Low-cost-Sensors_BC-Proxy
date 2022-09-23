#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBJECTIVE : This script adds noise to a data set and trains a denoising filter from the training set
This denoised data set is applied to the noisy testing set and then this filtered data set is used 
to estimate BC concentration.

Created on Fri Sep 16 10:44:40 2022

@author: jparedes
"""
# modules
import time
import os
import pickle

import numpy as np
from numpy.random import default_rng
import pandas as pd

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

import BCP_DataSet as DS
import BCP_PreProcessing as PP
import BCP_LCS_Cal as CAL

#%%
def proxy_data_set(Ref_BC, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N,freq='10min'):
    """
    Creates the data set of predictors that will participate in the BC proxy model
    Not all the measurements are at the same time instant or have the same time frequency. 
    The data sets must be averaged to the worst frequency.
    
    The BC measurements are located in the first column

    Parameters
    ----------
    Ref_BC : pandas DataFrame
        Reference Station BC measurements
    Ref_Meteo : pandas DataFrame
        Reference station T and RH measurements
    LCS_O3 : pandas DataFrame
        LCS O3 non-calibrated measurements
    LCS_NO2 : pandas DataFrame
        LCS NO2 non-calibrated measurements
    LCS_NO : pandas DataFrame
        LCS NO non-calibrated measurements
    LCS_Meteo : pandas DataFrame
        LCS inner device T and RH
    LCS_PM1 : pandas DataFrame
        LCS PM1 non-calibrated measurements
    LCS_PM25 : pandas DataFrame
        LCS PM2.5 non-calibrated measurements
    LCS_PM10 : pandas DataFrame
        LCS PM10 non-calibrated measurements
    LCS_N : pandas DataFrame
        LCS UFP non-calibrated measurements
    LCS_O3_cal : pandas DataFrame
        LCS O3 calibrated measurements
    LCS_NO2_cal : pandas DataFrame
        LCS NO2 calibrated measurements
    LCS_NO_cal : pandas DataFrame
        LCS NO calibrated measurements
    LCS_PM10_cal : pandas DataFrame
        LCS PM10 calibrated measurements
    freq : str
        frequency to average data set measurements, default is 10min

    Returns
    -------
    ds : pandas DataFrame
        data set of BC and its predictors for proxy model

    """
    # average data sets to the worst frequency - calibrated data sets are already averaged
    print('Averaging data sets every %s'%freq)
    # target
    Ref_BC = Ref_BC.groupby(pd.Grouper(key='date',freq=freq)).mean()
    # lcs predictors
    LCS_O3 = LCS_O3.groupby(pd.Grouper(key='date',freq=freq)).mean()
    LCS_NO2 = LCS_NO2.groupby(pd.Grouper(key='date',freq=freq)).mean()
    LCS_NO = LCS_NO.groupby(pd.Grouper(key='date',freq=freq)).mean()
    LCS_Meteo = LCS_Meteo.groupby(pd.Grouper(key='date',freq=freq)).mean()
    LCS_PM10 = LCS_PM10.groupby(pd.Grouper(key='date',freq=freq)).mean()
    LCS_PM1 = LCS_PM1.groupby(pd.Grouper(key='date',freq=freq)).mean()
    LCS_PM25 = LCS_PM25.groupby(pd.Grouper(key='date',freq=freq)).mean()
    LCS_N = LCS_N.groupby(pd.Grouper(key='date',freq=freq)).mean()
    # meteorological measurements are already every 10min
    Ref_Meteo.set_index('date',inplace=True)
    
    print('Concatenating data')
    ds = pd.concat([Ref_BC,LCS_O3,LCS_NO2,LCS_NO,LCS_Meteo,LCS_PM10,LCS_PM1,LCS_PM25,LCS_N,Ref_Meteo],axis=1)
    # delete non-matching times
    ds.dropna(inplace=True)
    
    return ds

def proxy_train_test_split(ds,test_frac=0.25):
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

#%%
def train_noise_filter(X_train):
    """
    Creates a filter object from training data
    Currently, the filter consists on a PCA

    Parameters
    ----------
    X_train : pandas DataFrame
                    noisy training data set 

    Returns
    -------
    noise_decomp : sklearn object
                    PCA based on SVD decomposition from training         

    """
    noise_decomp = PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, random_state=92)
    noise_decomp.fit(X_train)
    return noise_decomp    

def noise_add(X_train,X_test,seed=92,predictors=['WE_O3','AE_O3']):
    """
    Adds noise to training and testing data. 
    The noise is computed for different SNR values (>=1)
    
    SNR = E[X^2]/sigma_noise^2
    sigma_noise^2 = E[X^2]/SNR
    
    Noise is sampled from ~ N(0,sigma_noise^2)

    Parameters
    ----------
    X_test  : pandas DataFrame
                predictors training set

    X_test  : pandas DataFrame
                predictors testing set
                
    seed    : int
               random seed generator for sampling
               
    predictors : array
                array with noisy predictors names in it, default is O3 example: working and auxiliary electrode

    Returns
    -------
    X_train_noisy : list
                    predictors training set with noise added for different SNR, 
                    from lower to higher SNR

    """
    print('------------------------\nAdding noise to data set\n------------------------')
    X_train_ = X_train[predictors].copy()
    X_test_ = X_test[predictors].copy()
    df = X_test.copy()
    # compute SNR
    sp = np.mean( X_train_**2 ) # Signal Power
    
    snr_values_dB = np.linspace(0,35,8) # range of SNR values in dB scale
    X_test_noisy = []
    
    for snr_db in snr_values_dB: 
        snr = 10**(snr_db/10)
        std_n = (sp/snr)**0.5 # Noise std. deviation
        # sampling from noise distributions
        rng = default_rng(seed=seed)
        noise_sample = rng.normal(loc=0.0,scale=std_n,size=(X_test_.shape[0],len(predictors)))
                
        # adding noise to testing data
        noise_added = X_test_ + noise_sample
        noise_added[noise_added<0] = 0.0
        #noise_added.RH[noise_added['RH']>100]=100
        df[predictors] = noise_added
        # save data set with added noise
        X_test_noisy.append(df.copy())
        
    print('Done\n----')
        
        
      
      
    return X_test_noisy
#%%
def reconstruct_proxy_dataset(X_test):
    """
    Converts raw data set into data frame that can be used for calibration and proxy steps

    Parameters
    ----------
    X_test : pandas DataFrame
            raw testing set data set

    Returns
    -------
    X_test : pandas DataFrame
            data set containing merged data for O3 and NO2. Necessary for BC proxy model

    """
    # convert working and auxiliary electrodes readings into  signal
    X_test['S_NO2'] = X_test.loc[:,'WE_NO2']-X_test.loc[:,'AE_NO2'] 
    X_test['S_O3'] = (X_test.loc[:,'WE_O3']-X_test.loc[:,'AE_O3'])-X_test.loc[:,'S_NO2']
        
    return X_test

def calibrate_measurements(model_cal_O3_SVR,model_cal_NO2_SVR,X_test):
    """
    Calibrates lcs measurements

    Parameters
    ----------
    X_test : pandas DataFrame
        testing set data set containing lcs uncalibrated measurements
    model_cal_O3_SVR : sklearn pipeline
        lcs calibration model for O3
    model_cal_NO2_SVR : sklearn pipeline
        lcs calibration model for NO2

    Returns
    -------
    X_test : pandas dataFrame
        testing data set containing calibrated measurements columns

    """
    
    df_cal = X_test.copy()
    df_O3_lcs_cal = pd.DataFrame(model_cal_O3_SVR.predict(df_cal.loc[:,model_cal_O3_SVR.feature_names_in_]),index=df_cal.index,columns=['O3_cal_SVR'])
    df_NO2_lcs_cal = pd.DataFrame(model_cal_NO2_SVR.predict(df_cal.loc[:,model_cal_NO2_SVR.feature_names_in_]),index=df_cal.index,columns=['NO2_cal_SVR'])
    X_test['O3_cal_SVR'] = df_O3_lcs_cal
    X_test['NO2_cal_SVR'] = df_NO2_lcs_cal
    
    return X_test
    
    
    
    
#%%
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
def plot_noise_results(snr_values_dB=np.linspace(0,35,8), RMSE_orig = 0.442, R2_orig=0.7827,adjR2_orig=0.7821,save_fig=True):
    """
    Loads all BC proxy performance files of denoised testing data set
    10 different iterations of noise added. Seed runs from 0 to 9, both included
    The results are obtained from the optimal subspace reconstruction
    
    
    Parameters
    ----------
    snr_values: np.ndarray
                array of SNR in dB values used for noise/denoising algorithm
                
    RMSE_orig : float
                original RMSE BC proxy scoring for original testing set
                
    R2_orig : float
              original R2 BC proxy scoring for original testing set
    
    ajR2_orig : float
                original adjusted R2 BC proxy scoring for original testing set
    
    Returns
    -------
    df_RMSE : pandas DataFrame
        minimum RMSE from PCA reconstruction for different iteration of added noise
            
    df_R2 : pandas DataFrame
        minimum R2 from PCA reconstruction for different iteration of added noise
        
    df_adjR2 : pandas DataFrame
        minimum adj-R2 from PCA reconstruction for different iteration of added noise
        
    RMSE_noisy : pandas DataFrame
        RMSE of noisy data set for different SNR
        
    R2_noisy : pandas DataFrame
        R2 of noisy data set for different SNR
    
    adjR2_noisy : pandas DataFrame
        adjusted R2 of noisy data set for different SNR
    
    fig : matplotlib figure
            figure showing RMSE,R2 and adjusted R2 scores for different levels of SNR in dB
    """
    
    seed = np.arange(0,10,1)
    df_RMSE = pd.DataFrame()
    df_R2 = pd.DataFrame()
    df_adjR2 = pd.DataFrame()
    for s in seed:#load iterations of seed: optimal metric value is selected independent of k-value
            fname = 'RMSE_SNR_seed'+str(s)+'.pkl'
            df = pd.read_pickle(fname)
            df_RMSE[s] = df.min(axis=0) #minimum value for each SNR
            fname = 'R2_SNR_seed'+str(s)+'.pkl'
            df = pd.read_pickle(fname)
            df_R2[s] = df.max(axis=0) #maximum value for each SNR
            fname = 'adjR2_SNR_seed'+str(s)+'.pkl'
            df = pd.read_pickle(fname) #maximum value for each SNR
            df_adjR2[s] = df.max(axis=0)
    
    # load noisy results
    fname = 'RMSE_noisy.pkl'
    RMSE_noisy = pd.read_pickle(fname)
    fname = 'R2_noisy.pkl'
    R2_noisy = pd.read_pickle(fname)
    fname = 'adjR2_noisy.pkl'
    adjR2_noisy = pd.read_pickle(fname)
    
    #format
    df_RMSE.index = snr_values_dB
    df_R2.index = snr_values_dB
    df_adjR2.index = snr_values_dB
    RMSE_noisy.index = snr_values_dB
    R2_noisy.index = snr_values_dB
    adjR2_noisy.index = snr_values_dB
        
    
    
    ##plot
    # The plot is RMSE/R2/adjR2 vs SNR(dB)
    # There are different iterations for different seed values that are averaged
    
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(131)
    ax1.plot(snr_values_dB,df_RMSE.mean(axis=1),marker='^',color=(0.27,0.51,0.71),markersize=10,
             markeredgecolor='k',label='RMSE after denoising filter')
    ax1.fill_between(x=snr_values_dB,y1=df_RMSE.mean(axis=1)-df_RMSE.std(axis=1),
                     y2=df_RMSE.mean(axis=1)+df_RMSE.std(axis=1),color=(0.53,0.81,0.92))
    
    ax1.plot(snr_values_dB,RMSE_noisy.mean(axis=1),marker='^',color=(0,0.55,0.55),markersize=10,
             markeredgecolor='k',label='RMSE of noisy data set')
    ax1.fill_between(x=snr_values_dB,y1=RMSE_noisy.mean(axis=1)-RMSE_noisy.std(axis=1),
                     y2=RMSE_noisy.mean(axis=1)+RMSE_noisy.std(axis=1),color=(0.37,0.62,0.63))
    
    ax1.axhline(y=RMSE_orig, color='b', linestyle='--',label='No added noise')
    
  
    ymin = 0.4
    ymax = 1.2
    yrange=np.arange(ymin,ymax,0.2)
    
    ax1.set_yticks(ticks=yrange)
    ylabels = [str(np.round(i,2)) for i in yrange]
    ax1.set_yticklabels(labels=ylabels,fontsize=20)
    ax1.set_ylabel('RMSE [$\mu$g/$m^{3}$]',fontsize=20,rotation=90)
  
    xrange = snr_values_dB
    ax1.set_xticks(ticks=xrange)
    ax1.set_xticklabels(labels=[str(int(i)) for i in xrange],fontsize=20)
    ax1.set_xlabel('SNR$_{dB}$',fontsize=20)
  
    ax1.legend(loc='upper right',prop={'size':15},ncol=1,framealpha=0.4)
    ax1.grid(alpha=0.5)
      
  
    ax2 = fig.add_subplot(132)
  
    ax2.plot(snr_values_dB,df_R2.mean(axis=1),marker='^',color=(0.72,0.53,0.04),markersize=10,
             markeredgecolor='k',label='R$^2$ after denoising filter')
    ax2.fill_between(x=snr_values_dB,y1=df_R2.mean(axis=1)-df_R2.std(axis=1),
                     y2=df_R2.mean(axis=1)+df_R2.std(axis=1),color=(0.94,0.9,0.55))
    
    ax2.plot(snr_values_dB,R2_noisy.mean(axis=1),marker='^',color=(1.,0.65,0),markersize=10,
             markeredgecolor='k',label='R$^2$ of noisy data set')
    ax2.fill_between(x=snr_values_dB,y1=R2_noisy.mean(axis=1)-R2_noisy.std(axis=1),
                     y2=R2_noisy.mean(axis=1)+R2_noisy.std(axis=1),color=(0.91,0.59,0.48))
    
    
    ax2.axhline(y=R2_orig, color='r', linestyle='--',label='No added noise')
    
    ymin = -0.2
    ymax = 1.0
    yrange=np.arange(ymin,ymax,0.2)
    
    ax2.set_yticks(ticks=yrange)
    ylabels = [str(np.round(i,2)) for i in yrange]
    ax2.set_yticklabels(labels=ylabels,fontsize=20)
    ax2.set_ylabel('$R^2$',fontsize=20,rotation=90)
  
    xrange = snr_values_dB
    ax2.set_xticks(ticks=xrange)
    ax2.set_xticklabels(labels=[str(int(i)) for i in xrange],fontsize=20)
    ax2.set_xlabel('SNR$_{dB}$',fontsize=20)
  
    ax2.legend(loc='lower right',prop={'size':15},ncol=1,framealpha=0.4)
    ax2.grid(alpha=0.5)
    
    
    ax3 = fig.add_subplot(133)
    ax3.plot(snr_values_dB,df_adjR2.mean(axis=1),marker='^',color=(0,.5,0),markersize=10,
             markeredgecolor='k',label='adj-R$^2$ after denoising filter')
    ax3.fill_between(x=snr_values_dB,y1=df_adjR2.mean(axis=1)-df_adjR2.std(axis=1),
                     y2=df_adjR2.mean(axis=1)+df_adjR2.std(axis=1),color=(.56,0.93,.56))
    ax3.plot(snr_values_dB,adjR2_noisy.mean(axis=1),marker='^',color=(.51,.51,0),markersize=10,
             markeredgecolor='k',label='adj-R$^2$ of noisy data set')
    ax3.fill_between(x=snr_values_dB,y1=adjR2_noisy.mean(axis=1)-adjR2_noisy.std(axis=1),
                     y2=adjR2_noisy.mean(axis=1)+adjR2_noisy.std(axis=1),color=(.77,.77,0))
    ax3.axhline(y=adjR2_orig, color='r', linestyle='--',label='No added noise')
    
    
    ymin = -0.2
    ymax = 1.0
    yrange=np.arange(ymin,ymax,0.2)
    
    ax3.set_yticks(ticks=yrange)
    ylabels = [str(np.round(i,2)) for i in yrange]
    ax3.set_yticklabels(labels=ylabels,fontsize=20)
    ax3.set_ylabel('adjusted $R^2$',fontsize=20,rotation=90)
  
    xrange = snr_values_dB
    ax3.set_xticks(ticks=xrange)
    ax3.set_xticklabels(labels=[str(int(i)) for i in xrange],fontsize=20)
    ax3.set_xlabel('SNR$_{dB}$',fontsize=20)
  
    ax3.legend(loc='lower right',prop={'size':15},ncol=1,framealpha=0.4)
    ax3.grid(alpha=0.5)
  
  
    plt.suptitle('Black carbon proxy results\n after testing set denoising\n',fontsize=35)
    
    for axis in ['top','bottom','left','right']:
      ax1.spines[axis].set_linewidth(2)
      ax2.spines[axis].set_linewidth(2)
      ax3.spines[axis].set_linewidth(2)
  
    fig.tight_layout()
    if save_fig:
        plt.savefig("BC_proxy_denoising_scores.png", bbox_inches='tight', dpi=600)
    
    
    return df_RMSE,df_R2,df_adjR2,RMSE_noisy,R2_noisy,adjR2_noisy,fig

def plot_denoising_changing_predictors(snr_values_dB=np.linspace(0,35,8), RMSE_orig = 0.442, R2_orig=0.7827,adjR2_orig=0.7821,save_fig=True):
    
    seed = np.arange(0,10,1)
    df_RMSE_O3 = pd.DataFrame()
    df_R2_O3 = pd.DataFrame()
    path = 'O3_noisy'
    os.chdir(path)
    
    for s in seed:
            fname = 'RMSE_SNR_seed'+str(s)+'.pkl'
            df = pd.read_pickle(fname)
            df_RMSE_O3[s] = df.min(axis=0) #minimum value for each SNR
            fname = 'R2_SNR_seed'+str(s)+'.pkl'
            df = pd.read_pickle(fname)
            df_R2_O3[s] = df.max(axis=0) #maximum value for each SNR
            
    
    # load noisy results
    fname = 'RMSE_noisy.pkl'
    RMSE_noisy_O3 = pd.read_pickle(fname)
    fname = 'R2_noisy.pkl'
    R2_noisy_O3 = pd.read_pickle(fname)
    os.chdir('..')
    
    #format
    df_RMSE_O3.index = snr_values_dB
    df_R2_O3.index = snr_values_dB
    
    RMSE_noisy_O3.index = snr_values_dB
    R2_noisy_O3.index = snr_values_dB
        
    # NO2
    
    path = 'NO2_noisy'
    os.chdir(path)
    df_RMSE_NO2 = pd.DataFrame()
    df_R2_NO2 = pd.DataFrame()
    
    for s in seed:
            fname = 'RMSE_SNR_seed'+str(s)+'.pkl'
            df = pd.read_pickle(fname)
            df_RMSE_NO2[s] = df.min(axis=0) #minimum value for each SNR
            fname = 'R2_SNR_seed'+str(s)+'.pkl'
            df = pd.read_pickle(fname)
            df_R2_NO2[s] = df.max(axis=0) #maximum value for each SNR
            
    
    # load noisy results
    fname = 'RMSE_noisy.pkl'
    RMSE_noisy_NO2 = pd.read_pickle(fname)
    fname = 'R2_noisy.pkl'
    R2_noisy_NO2 = pd.read_pickle(fname)
    
    
    #format
    df_RMSE_NO2.index = snr_values_dB
    df_R2_NO2.index = snr_values_dB
    
    RMSE_noisy_NO2.index = snr_values_dB
    R2_noisy_NO2.index = snr_values_dB
    
    os.chdir('..')
    
    # N0.3
    path = 'N_03_noisy'
    os.chdir(path)
    
    df_RMSE_N03 = pd.DataFrame()
    df_R2_N03 = pd.DataFrame()
    
    for s in seed:
            fname = 'RMSE_SNR_seed'+str(s)+'.pkl'
            df = pd.read_pickle(fname)
            df_RMSE_N03[s] = df.min(axis=0) #minimum value for each SNR
            fname = 'R2_SNR_seed'+str(s)+'.pkl'
            df = pd.read_pickle(fname)
            df_R2_N03[s] = df.max(axis=0) #maximum value for each SNR
            
    
    # load noisy results
    fname = 'RMSE_noisy.pkl'
    RMSE_noisy_N03 = pd.read_pickle(fname)
    fname = 'R2_noisy.pkl'
    R2_noisy_N03 = pd.read_pickle(fname)
    
    
    #format
    df_R2_N03.index = snr_values_dB
    df_R2_N03.index = snr_values_dB
    
    RMSE_noisy_N03.index = snr_values_dB
    R2_noisy_N03.index = snr_values_dB
    
    os.chdir('..')
    
    # N 1.0
    path = 'N_1_noisy'
    os.chdir(path)
    
    df_RMSE_N1 = pd.DataFrame()
    df_R2_N1 = pd.DataFrame()
    
    for s in seed:
            fname = 'RMSE_SNR_seed'+str(s)+'.pkl'
            df = pd.read_pickle(fname)
            df_RMSE_N1[s] = df.min(axis=0) #minimum value for each SNR
            fname = 'R2_SNR_seed'+str(s)+'.pkl'
            df = pd.read_pickle(fname)
            df_R2_N1[s] = df.max(axis=0) #maximum value for each SNR
            
    
    # load noisy results
    fname = 'RMSE_noisy.pkl'
    RMSE_noisy_N1 = pd.read_pickle(fname)
    fname = 'R2_noisy.pkl'
    R2_noisy_N1 = pd.read_pickle(fname)
    
    
    #format
    df_R2_N1.index = snr_values_dB
    df_R2_N1.index = snr_values_dB
    
    RMSE_noisy_N1.index = snr_values_dB
    R2_noisy_N1.index = snr_values_dB
    
    os.chdir('..')
    
    
    
    ##plot
    # The plot is RMSE/R2 vs SNR(dB)
    # There are different iterations for different seed values that are averaged
    
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(211)
    ax1.axhline(y=RMSE_orig, color='b', linestyle='--',label='No added noise')
    ax1.plot(snr_values_dB,df_RMSE_O3.mean(axis=1),marker='^',color=(0.27,0.51,0.71),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Denoised O$_3$')
    ax1.fill_between(x=snr_values_dB,y1=df_RMSE_O3.mean(axis=1)-df_RMSE_O3.std(axis=1),alpha=.35,
                     y2=df_RMSE_O3.mean(axis=1)+df_RMSE_O3.std(axis=1),color=(0.53,0.81,0.92))
    
    ax1.plot(snr_values_dB,RMSE_noisy_O3.mean(axis=1),marker='^',color=(0,0.55,0.55),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Noisy O$_3$')
    ax1.fill_between(x=snr_values_dB,y1=RMSE_noisy_O3.mean(axis=1)-RMSE_noisy_O3.std(axis=1),alpha=.35,
                     y2=RMSE_noisy_O3.mean(axis=1)+RMSE_noisy_O3.std(axis=1),color=(0.37,0.62,0.63))
    ## NO2
    
    ax1.plot(snr_values_dB,df_RMSE_NO2.mean(axis=1),marker='^',color=(0.72,0.53,0.04),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Denoised NO$_2$')
    ax1.fill_between(x=snr_values_dB,y1=df_RMSE_NO2.mean(axis=1)-df_RMSE_NO2.std(axis=1),alpha=.35,
                     y2=df_RMSE_NO2.mean(axis=1)+df_RMSE_NO2.std(axis=1),color=(0.94,0.9,0.55))
    
    ax1.plot(snr_values_dB,RMSE_noisy_NO2.mean(axis=1),marker='^',color=(1.,0.65,0),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Noisy NO$_2$')
    ax1.fill_between(x=snr_values_dB,y1=RMSE_noisy_NO2.mean(axis=1)-RMSE_noisy_NO2.std(axis=1),alpha=.35,
                     y2=RMSE_noisy_NO2.mean(axis=1)+RMSE_noisy_NO2.std(axis=1),color=(0.91,0.59,0.48))
    
    
    
  
    
    ## N 0.3
    ax1.plot(snr_values_dB,df_RMSE_N03.mean(axis=1),marker='^',color=(0,.5,0),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Denoised N$_{0.3}$')
    ax1.fill_between(x=snr_values_dB,y1=df_RMSE_N03.mean(axis=1)-df_RMSE_N03.std(axis=1),alpha=.35,
                     y2=df_RMSE_N03.mean(axis=1)+df_RMSE_N03.std(axis=1),color=(.56,0.93,.56))
    
    ax1.plot(snr_values_dB,RMSE_noisy_N03.mean(axis=1),marker='^',color=(.51,.51,0),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Noisy N$_{0.3}$')
    ax1.fill_between(x=snr_values_dB,y1=RMSE_noisy_N03.mean(axis=1)-RMSE_noisy_N03.std(axis=1),alpha=.35,
                     y2=RMSE_noisy_N03.mean(axis=1)+RMSE_noisy_N03.std(axis=1),color=(.77,.77,0))
    
    
    ## N 1.0
    ax1.plot(snr_values_dB,df_RMSE_N1.mean(axis=1),marker='^',color=(.5,0,.5),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Denoised N$_1$')
    ax1.fill_between(x=snr_values_dB,y1=df_RMSE_N1.mean(axis=1)-df_RMSE_N1.std(axis=1),alpha=.35,
                     y2=df_RMSE_N1.mean(axis=1)+df_RMSE_N1.std(axis=1),color=(.87,0.63,.87))
    
    ax1.plot(snr_values_dB,RMSE_noisy_N1.mean(axis=1),marker='^',color=(.5,0,0),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Noisy N$_1$')
    ax1.fill_between(x=snr_values_dB,y1=RMSE_noisy_N1.mean(axis=1)-RMSE_noisy_N1.std(axis=1),alpha=.35,
                     y2=RMSE_noisy_N1.mean(axis=1)+RMSE_noisy_N1.std(axis=1),color=(.86,.44,.58))
    
    ymin = 0.4
    ymax = 1.2
    yrange=np.arange(ymin,ymax,0.2)
    
    ax1.set_yticks(ticks=yrange)
    ylabels = [str(np.round(i,2)) for i in yrange]
    
    ax1.set_yticklabels(labels=ylabels,fontsize=27)
    ax1.set_ylabel('RMSE [$\mu$g/$m^{3}$]',fontsize=28,rotation=90)
  
    xrange = np.append(snr_values_dB,snr_values_dB[-1]+5)
    ax1.set_xticks(ticks=xrange)
    xlabel = [str(int(i)) for i in xrange]
    xlabel[-1] = ' '
    ax1.set_xticklabels(labels=xlabel,fontsize=20)
    ax1.tick_params(direction='out', length=4, width=1)
    ax1.tick_params(axis='both', which='major', labelsize=22)
    #ax1.set_xlabel('SNR$_{dB}$',fontsize=20)
  
    ax1.legend(loc='upper right',prop={'size':13},ncol=1,framealpha=0.8,edgecolor = 'black',handleheight = 1,handletextpad=0.2)
    ax1.grid(alpha=0.5)
    
    # R2
    ax2 = fig.add_subplot(212)
    ax2.axhline(y=R2_orig, color='r', linestyle='--',label='No added noise')
    ax2.plot(snr_values_dB,df_R2_O3.mean(axis=1),marker='^',color=(0.27,0.51,0.71),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Denoised O$_3$')
    ax2.fill_between(x=snr_values_dB,y1=df_R2_O3.mean(axis=1)-df_R2_O3.std(axis=1),alpha=.35,
                     y2=df_R2_O3.mean(axis=1)+df_R2_O3.std(axis=1),color=(0.53,0.81,0.92))
    
    ax2.plot(snr_values_dB,R2_noisy_O3.mean(axis=1),marker='^',color=(0,0.55,0.55),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Noisy O$_3$')
    ax2.fill_between(x=snr_values_dB,y1=R2_noisy_O3.mean(axis=1)-R2_noisy_O3.std(axis=1),alpha=.35,
                     y2=R2_noisy_O3.mean(axis=1)+R2_noisy_O3.std(axis=1),color=(0.37,0.62,0.63))
    
    ## NO2
    ax2.plot(snr_values_dB,df_R2_NO2.mean(axis=1),marker='^',color=(0.72,0.53,0.04),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Denoised NO$_2$')
    ax2.fill_between(x=snr_values_dB,y1=df_R2_NO2.mean(axis=1)-df_R2_NO2.std(axis=1),alpha=.35,
                     y2=df_R2_NO2.mean(axis=1)+df_R2_NO2.std(axis=1),color=(0.94,0.9,0.55))
    
    ax2.plot(snr_values_dB,R2_noisy_NO2.mean(axis=1),marker='^',color=(1.,0.65,0),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Noisy NO$_2$')
    ax2.fill_between(x=snr_values_dB,y1=R2_noisy_NO2.mean(axis=1)-R2_noisy_NO2.std(axis=1),alpha=.35,
                     y2=R2_noisy_NO2.mean(axis=1)+R2_noisy_NO2.std(axis=1),color=(0.91,0.59,0.48))
    
    
    
    ## N 0.3
    ax2.plot(snr_values_dB,df_R2_N03.mean(axis=1),marker='^',color=(0,.5,0),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Denoised N$_{0.3}$')
    ax2.fill_between(x=snr_values_dB,y1=df_R2_N03.mean(axis=1)-df_R2_N03.std(axis=1),alpha=.35,
                     y2=df_R2_N03.mean(axis=1)+df_R2_N03.std(axis=1),color=(.56,0.93,.56))
    
    ax2.plot(snr_values_dB,R2_noisy_N03.mean(axis=1),marker='^',color=(.51,.51,0),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Noisy N$_{0.3}$')
    ax2.fill_between(x=snr_values_dB,y1=R2_noisy_N03.mean(axis=1)-R2_noisy_N03.std(axis=1),alpha=.35,
                     y2=R2_noisy_N03.mean(axis=1)+R2_noisy_N03.std(axis=1),color=(.77,.77,0))
    
    ## N 1.0
    ax2.plot(snr_values_dB,df_R2_N1.mean(axis=1),marker='^',color=(.5,0,.5),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Denoised N$_{1}$')
    ax2.fill_between(x=snr_values_dB,y1=df_R2_N1.mean(axis=1)-df_R2_N1.std(axis=1),alpha=.35,
                     y2=df_R2_N1.mean(axis=1)+df_R2_N1.std(axis=1),color=(.87,0.63,.87))
    
    ax2.plot(snr_values_dB,R2_noisy_N1.mean(axis=1),marker='^',color=(.5,0,0),markersize=10,linewidth = 3.5,
             markeredgecolor='k',label='Noisy N$_{1}$')
    ax2.fill_between(x=snr_values_dB,y1=R2_noisy_N1.mean(axis=1)-R2_noisy_N1.std(axis=1),alpha=.35,
                     y2=R2_noisy_N1.mean(axis=1)+R2_noisy_N1.std(axis=1),color=(.86,.44,.58))
    

    
    
    
  
    ymin = 0.2
    ymax = 1.0
    yrange=np.arange(ymin,ymax,0.2)
    
    ax2.set_yticks(ticks=yrange)
    ylabels = [str(np.round(i,2)) for i in yrange]
    
    ax2.set_yticklabels(labels=ylabels,fontsize=27)
    ax2.set_ylabel('R$^2$',fontsize=28,rotation=90)
  
    xrange = np.append(snr_values_dB,snr_values_dB[-1]+5)
    ax2.set_xticks(ticks=xrange)
    xlabel = [str(int(i)) for i in xrange]
    xlabel[-1] = ' '
    ax2.set_xticklabels(labels=xlabel,fontsize=20)
    ax2.set_xlabel('SNR$_{dB}$',fontsize=26)
    ax2.tick_params(direction='out', length=4, width=1)
    ax2.tick_params(axis='both', which='major', labelsize=22)
  
    ax2.legend(loc='lower right',prop={'size':13},ncol=1,framealpha=0.8,edgecolor = 'black',handleheight = 1,handletextpad=0.2)
    
    ax2.grid(alpha=0.5)
    
    plt.suptitle('Black carbon proxy results\n after applying the noise reduction filter\n',fontsize=35)
    
    for axis in ['top','bottom','left','right']:
      ax1.spines[axis].set_linewidth(2)
      ax2.spines[axis].set_linewidth(2)
  
    fig.tight_layout()
    fig.set_size_inches(14,9)
    
    if save_fig:
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/4_Proxy_deployment/Noise_filter/raw_noisy'
        plt.savefig("BC_proxy_denoising_raw_lcs.png", bbox_inches='tight', dpi=600)

    
    return fig

def denoise_testing_set(X_test,noise_filter,scaler,model_cal_O3_SVR,model_cal_NO2_SVR,model_BC_proxy,Y_test,X_test_raw):
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/4_Proxy_deployment/Noise_filter/raw_noisy/non_noisy_testing'
    os.chdir(path)
    
    X_test_centered = pd.DataFrame(scaler.transform(X_test))
    
    print('\nApplying noise filter to noisy data\n')
    RMSE_filtered = []
    R2_filtered = []
    adj_R2_filtered = []
    filtered_pollutant = ['WE_NO2','AE_NO2']
    
    for k in range(1,noise_filter.shape[0]+1):#try different number of SVD components
        
        # apply the filter to the testing set
        print('\n\n\nConsidering %i vectors for projecting'%k)    
        U_matrix = noise_filter.iloc[:,0:k]
        Pr = U_matrix.T @ X_test_centered.T.to_numpy() #projection
        X_test_noise_filtered = U_matrix @ Pr #reconstruction
        # format
        X_test_filtered = X_test_noise_filtered.T
        X_test_filtered = pd.DataFrame(scaler.inverse_transform(X_test_filtered))
        X_test_filtered.columns = X_test.columns
        X_test_filtered.index = X_test.index
        print('-----------------\nFiltered data set\n-----------------')
        non_filtered_pollutants = [p for p in X_test.columns if p not in filtered_pollutant]
        X_test_filtered.loc[:,non_filtered_pollutants] = X_test.loc[:,non_filtered_pollutants]
        print(X_test_filtered)
        
        # Use the filtered data set for prediction
        ## reconstruct data set that model needs
        X_test_filtered = reconstruct_proxy_dataset(X_test_filtered)
        #X_test_filtered['T_int'] = X_test_raw.loc[:,'T_int']
        #X_test_filtered['RH_int'] = X_test_raw.loc[:,'RH_int']
        
        ## calibration
        X_test_filtered = calibrate_measurements(model_cal_O3_SVR,model_cal_NO2_SVR,X_test_filtered)
        
        
        print('\n--------------------------------------\nUsing filtered data set for prediction\n--------------------------------------\n')
        y_pred = model_BC_proxy.predict(X_test_filtered.loc[:,model_BC_proxy.feature_names_in_])
        # scoring metrics
        RMSE,R2,adj_R2 = scoring(Y_test,y_pred,X_test)
        print('Results for k=%i'%k)
        print('RMSE = %.2f\nR^2 = %.2f\n-----------'%(RMSE,R2))
        
        RMSE_filtered.append(RMSE)
        R2_filtered.append(R2)
        adj_R2_filtered.append(adj_R2)
        
        
    
    return RMSE_filtered,R2_filtered,adj_R2_filtered

#%%
def main():
    #%%
    ### -----------load dataset-----------
     
    #load data sets for every pollutant
    
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/1_Files/raw_data_files'
    print('Loading data set from %s'%path)
    Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = DS.load_dataSets(path)

    print('Pre-processing steps')
    Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = PP.pre_processing(Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N)
    ds_proxy = proxy_data_set(Ref_BC, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N,freq='10min')
    
    raw_predictors = ['BC','WE_O3','AE_O3','WE_NO2','AE_NO2','S5_0.3','S5_1.0','T','RH','T_int','RH_int']
    ds_noise = ds_proxy[raw_predictors]
    
    X_train_proxy,Y_train_proxy,X_test_proxy,Y_test_proxy = proxy_train_test_split(ds_noise,test_frac=0.25)


    #%% load calibration models
    print('\n\n--------------------------\nLoading calibration models\n--------------------------\n\n')
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/O3'
    os.chdir(path)
    model_cal_O3_MLR,model_cal_O3_SVR = CAL.load_calibration_models('O3')
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/NO2'
    os.chdir(path)
    model_cal_NO2_MLR,model_cal_NO2_SVR = CAL.load_calibration_models('NO2')
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/NO'
    os.chdir(path)
    model_cal_NO_MLR,model_cal_NO_SVR = CAL.load_calibration_models('NO')
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/2_Calibration/PM10'
    os.chdir(path)
    model_cal_PM10_MLR,model_cal_PM10_SVR = CAL.load_calibration_models('PM10')
    
    #%% load BC proxy model
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/3_BC_Proxy/Final_model'
    os.chdir(path)
    fname = 'BC_proxy_FinalModel.pkl'
    print('Loading BC proxy model')
    model_BC_proxy = joblib.load(fname)
   
    
    #%%
    ########################################################
    # Denoising filter test
    # 1. Learn denoise filter from training data
    # 2. Add noise to testing data
    # 3. Apply denoise filter to noisy testing data
    # 4. Calibrate filtered testing data
    # 5. BC proxy prediction using calibrated filtered data
    ########################################################
    print('Simulating noise on testing set predictors and denoising filter for predicting BC concentration during deployment')
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/4_Proxy_deployment/Noise_filter'
    os.chdir(path)
    print('Changing directory\n%s'%path)

    # learn  noise filter from training data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_proxy))
    noise_filter = pd.DataFrame(train_noise_filter(X_train_scaled).components_.T)# projection matrix
    singular_values = train_noise_filter(X_train_scaled).singular_values_
    save_filter = False
    if save_filter:
        print('Saving Noise reduction filter obtained from raw training set')
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/4_Proxy_deployment/Noise_filter/raw_noisy'
        os.chdir(path)
        fname = 'denoise_matrix.pkl'
        noise_filter.to_pickle(fname)
        fname = 'denoise_pca.pkl'
        joblib.dump(train_noise_filter(X_train_scaled),fname)
        
        
    

    # simulate noise on testing set: different SNR for different iterations
    RMSE_unfiltered = pd.DataFrame()
    R2_unfiltered = pd.DataFrame()
    adj_R2_unfiltered = pd.DataFrame()
    noisy_predictors = ['S5_1.0']
    print('Adding noise to raw predictors %s'%[i for i in noisy_predictors])
    for s in range(10):#repeat for different added noises
    
        # add noise to all predictors
        print('\n------------\nAdding noise\n------------\n')
        X_test_noisy = noise_add(X_train_proxy,X_test_proxy,seed=92+s,predictors=noisy_predictors)
        print('%i different SNR values considered'%len(X_test_noisy))
        # denoised data sets BC proxy scoring metrics
        RMSE_snr = pd.DataFrame()
        R2_snr = pd.DataFrame()
        adj_R2_snr = pd.DataFrame()
        # noisy data sets BC proxy scoring emtrics
        RMSE_noisy = []
        R2_noisy = []
        adj_R2_noisy = []
        
        for i in range(len(X_test_noisy)):# for every SNR
            X_test_noisy_scaled = pd.DataFrame(scaler.transform(X_test_noisy[i]))
            X_test_noisy_centered = X_test_noisy_scaled#-X_train_scaled.mean(axis=0)  #if data sets aqre scaled then mean is 0
            print('\nApplying noise filter to noisy data\n')
            RMSE_filtered = []
            R2_filtered = []
            adj_R2_filtered = []
            noise_variables = [j for j in X_test_noisy[i].columns]

            
            ## Denoising: k is not settled in advance. Iterate over all possibilities
            for k in range(1,len(singular_values)+1):#try different number of SVD components
                # apply the filter to the testing set
                print('Considering %i vectors for projecting'%k)    
                U_matrix = noise_filter.iloc[:,0:k]
                Pr = U_matrix.T @ X_test_noisy_centered.T.to_numpy() #projection
                X_test_noise_filtered = U_matrix @ Pr #reconstruction
                # format
                X_test_filtered = X_test_noise_filtered.T
                X_test_filtered = pd.DataFrame(scaler.inverse_transform(X_test_filtered))
                X_test_filtered.columns = X_train_proxy.columns
                X_test_filtered.index = X_test_proxy.index
                ## correct wrong assignment to non noisy predictors fiven the sub-space projection
                non_noisy_predictors = [p for p in noise_variables if p not in noisy_predictors]
                X_test_filtered.loc[:,non_noisy_predictors] = X_test_noisy[i].loc[:,non_noisy_predictors]
                
                #X_test_filtered += X_train.mean(axis=0)#not mandatory if data set was scaled
                print('-----------------\nFiltered data set\n-----------------')
                print(X_test_filtered)
                
                # Use the filtered data set for prediction
                ## reconstruct data set that model needs
                X_test_filtered = reconstruct_proxy_dataset(X_test_filtered)
                ## calibration
                X_test_filtered = calibrate_measurements(model_cal_O3_SVR,model_cal_NO2_SVR,X_test_filtered)
                
                print('\n--------------------------------------\nUsing filtered data set for prediction\n--------------------------------------\n')
                y_pred = model_BC_proxy.predict(X_test_filtered.loc[:,model_BC_proxy.feature_names_in_])
                # scoring metrics
                RMSE,R2,adj_R2 = scoring(Y_test_proxy,y_pred,X_test_proxy)
                RMSE_filtered.append(RMSE)
                R2_filtered.append(R2)
                adj_R2_filtered.append(adj_R2)
            
            
            ## save prediction scoring for different SNR values
            # format
            RMSE_filtered = pd.DataFrame(RMSE_filtered)
            R2_filtered = pd.DataFrame(R2_filtered)
            adj_R2_filtered = pd.DataFrame(adj_R2_filtered)
           
            RMSE_snr[i] = RMSE_filtered
            R2_snr[i] = R2_filtered
            adj_R2_snr[i] = adj_R2_filtered
            
            
            ## predict using the noisy & unfiltered data set for comparisson
            print('-------------------------------\nPredict BC using noisy data set\n-------------------------------\n')
            df = reconstruct_proxy_dataset(X_test_noisy[i])
            df_ = calibrate_measurements(model_cal_O3_SVR,model_cal_NO2_SVR,df)
            y_pred = model_BC_proxy.predict(df_.loc[:,model_BC_proxy.feature_names_in_])
            #scoring metrics
            RMSE,R2,adj_R2 = scoring(Y_test_proxy,y_pred,X_test_proxy)
            RMSE_noisy.append(RMSE)
            R2_noisy.append(R2)
            adj_R2_noisy.append(adj_R2)
            
            
            
        ## format denoised results
        # each column represents snr_values entry
        # each row represents the rank of the sub-space
        RMSE_snr.columns = [i for i in range(1,RMSE_snr.shape[1]+1)]
        RMSE_snr.index = [i for i in RMSE_snr.index+1]
        R2_snr.columns = [i for i in range(1,R2_snr.shape[1]+1)]
        R2_snr.index = [i for i in R2_snr.index+1]
        adj_R2_snr.columns = [i for i in range(1,adj_R2_snr.shape[1]+1)]
        adj_R2_snr.index = [i for i in adj_R2_snr.index+1]
        
        # save denoised results for a given noise seed
        print('Saving Results for this added noise to %s'%path)            
        fname = 'RMSE_SNR_seed'+str(s)+'.pkl'
        RMSE_snr.to_pickle(fname)
        fname = 'R2_SNR_seed'+str(s)+'.pkl'
        R2_snr.to_pickle(fname)
        fname = 'adjR2_SNR_seed'+str(s)+'.pkl'
        adj_R2_snr.to_pickle(fname)
        
        ## format noisy results
        # each column represents SNR values
        # each row represents seed
        RMSE_noisy = pd.DataFrame(RMSE_noisy)
        R2_noisy = pd.DataFrame(R2_noisy)
        adj_R2_noisy = pd.DataFrame(adj_R2_noisy)
        
        RMSE_unfiltered[s] = RMSE_noisy
        R2_unfiltered[s] = R2_noisy
        adj_R2_unfiltered[s] = adj_R2_noisy
        
    # save noisy results
    fname = 'RMSE_noisy.pkl'
    RMSE_unfiltered.to_pickle(fname)
    fname = 'R2_noisy.pkl'
    R2_unfiltered.to_pickle(fname)
    fname = 'adjR2_noisy.pkl'
    adj_R2_unfiltered.to_pickle(fname)
    print('\n\n--------------------------\nDenoising process finished\n--------------------------\n\n')
    
    
    
    
    # Once results are obtained: load files and plot
    RMSE,R2,adjR2,RMSE_noisy,R2_noisy,adjR2_noisy,fig_SNR = plot_noise_results(save_fig=True)
    
    return X_train_proxy,X_test_proxy,Y_train_proxy,Y_test_proxy,RMSE,R2,adjR2,RMSE_noisy,R2_noisy,adjR2_noisy,fig_SNR


#%%
if __name__ == '__main__':
    print('------------------------\nTesting denoising filter\n------------------------')
    compute_denoising = False
    if compute_denoising:
        X_train_proxy,X_test_proxy,Y_train_proxy,Y_test_proxy,RMSE,R2,adjR2,RMSE_noisy,R2_noisy,adjR2_noisy,fig_SNR = main()
    else:
        print('Plotting denoise filter results for different noisy predictors')
        path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/4_Proxy_deployment/Noise_filter/raw_noisy'
        os.chdir(path)
        fig = plot_denoising_changing_predictors(snr_values_dB=np.linspace(0,35,8), RMSE_orig = 0.442, R2_orig=0.7827,adjR2_orig=0.7821,save_fig=False)
        
    
