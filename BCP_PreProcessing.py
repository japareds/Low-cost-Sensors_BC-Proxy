#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OBJECTIVE: 
    This code performs data pre-processing.
    1) Time aggregation
    2) Keep relevant dates
    3) Missing values
    
Created on Mon Jun 27 11:57:07 2022

@author: jparedes
"""
import pandas as pd
import BCP_DataSet as DS
import numpy as np
#%%
"""
Pre-processing functions
"""

def bad_vals(df,pollutant):
    """
    Remove bad measurements (< 0)
    """
    print('Removing wrong measurements for '+str(pollutant)+'\n')
    df = df[df[pollutant]>0.0]
    return df

def Time_agg(df,time_period='10min'):
    """
    Time aggregation
    Each sensor dataset has different sample frequencies:

      BC time frequency: 1 min
      Reference Station time frequency: 10 min
      LCS Captor time frequency 1 s    
      LCS PM and UFP time frequency: 2 min
      
    In order to estimate BC, all the predictors
    must have measurements at the same time
    
    time_period: frequency for averaging
    """
    
    print('\n Aggregating data set every '+str(time_period))
    ### compute mean every t minutes
    df = df.groupby(pd.Grouper(key='date',freq=time_period)).mean()

    return df


def Time_period(df,start='2021-10-19 08:20:00',stop='2021-12-25 15:00:00',time_period='10min'):
    """
    This function defines a time interval and keeps information 
    only on that period of time
    
    start: starting date given by PM LCS
    stop: final date given by BC reference station
    time_period: freq of sampling
    """
    print('Keeping values for period '+start+' to '+stop+' at frequency of '+time_period)
    freq = time_period[0:2]+'T'
    time_range = pd.date_range(start,stop,freq=freq)
    df = df[df.index.isin(time_range)]
    return df


    

#%%
def pre_processing(Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N,time_period = '10min'):
    """
    pre processing step
    1. delete wrong measurements
    2. time aggregation (mean)
    3. keep relevant time period

    Parameters
    ----------
    Ref_BC : pandas DataFrame
        data set of Reference Station BC measurements.
    Ref_O3 : pandas DataFrame
        data set of Reference Station O3 measurements.
    Ref_NO2 : pandas DataFrame
        data set of Reference Station NO2 measurements.
    Ref_NO : pandas DataFrame
        data set of Reference Station NO measurements.
    Ref_PM10 : pandas DataFrame
        data set of Reference Station PM10 measurements.
    Ref_N : pandas DataFrame
        data set of Reference Station UFP measurements.
    Ref_Meteo : pandas DataFrame
        data set of Reference Station T and RH measurements.
    LCS_O3 : pandas DataFrame
        data set of LCS O3 measurements.
    LCS_NO2 : pandas DataFrame
        data set of LCS NO2 measurements.
    LCS_NO : pandas DataFrame
        data set of LCS NO measurements.
    LCS_Meteo : pandas DataFrame
        data set of LCS Internal T and RH measurements.
    LCS_PM1 : pandas DataFrame
        data set of LCS PM1 measurements.
    LCS_PM25 : pandas DataFrame
        data set of LCS PM 25 measurements.
    LCS_PM10 : pandas DataFrame
        data set of LCS PM10 measurements.
    LCS_N : pandas DataFrame
        data set of LCS UFP measurements.
    time_period : str, optional
        time frequency for aggregating measurements. The default is '10min'.

    Returns
    -------
    All data frames

    """
    ### remove wrong values
    
    # for reference station measurements
    Ref_BC = bad_vals(Ref_BC,Ref_BC.columns[1])
    Ref_O3 = bad_vals(Ref_O3,Ref_O3.columns[1])
    Ref_NO2 = bad_vals(Ref_NO2,Ref_NO2.columns[1])
    Ref_NO = bad_vals(Ref_NO,Ref_NO.columns[1])
    Ref_PM10 = bad_vals(Ref_PM10,Ref_PM10.columns[1])
    Ref_Meteo = bad_vals(Ref_Meteo,Ref_Meteo.columns[1])
    Ref_Meteo = bad_vals(Ref_Meteo,Ref_Meteo.columns[2])
    # for LCS internal Temperature and RH
    LCS_Meteo = bad_vals(LCS_Meteo,LCS_Meteo.columns[1])
    LCS_Meteo = bad_vals(LCS_Meteo,LCS_Meteo.columns[2])
    # for LCS PM and UFP    
    LCS_PM1 = bad_vals(LCS_PM1,LCS_PM1.columns[1])
    LCS_PM25 = bad_vals(LCS_PM25,LCS_PM25.columns[1])
    LCS_PM10 = bad_vals(LCS_PM10,LCS_PM10.columns[1])
    LCS_N = bad_vals(LCS_N,LCS_N.columns[1])
    
    ### Time average    
    # average data set every 10 min: lowest frequency
    #Ref_BC,Ref_O3,Ref_NO2,Ref_NO,Ref_PM10,Ref_N,Ref_Meteo,LCS_O3,LCS_NO2,LCS_NO,LCS_Meteo,LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = Time_agg(Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N,time_period=time_period)
    
    ### Relevant dates
    #df = Time_period(df,time_period=time_period)
    
    ### separate LCS and Ref. Station data sets
    #df_ref = df.iloc[:,0:8]
    #df_lcs = df.iloc[:,np.r_[0,6:df.shape[1]]]
    
    ### remove NaN
    #df_ref.dropna(inplace=True)
    #df_lcs.dropna(inplace=True)
    
    print('Pre-processing finished')
    return Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N
#%%
def main():
    """
    Test Pre-processing

    Returns
    -------
    df : pandas DataFrame
        data set with time cleaned, averaged pollutants for
        the relevant time period

    """
    ######################
    #-- Previous modules--
    ######################
    ### directory where pkl raw data set files are located
    path = '/home/jparedes/Documents/PhD/Files/Data/Proxy_LCS/1_Files/raw_data_files'
    Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = DS.load_dataSets(path)
    ######################
    #-pre-processing step-
    ######################
    Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = pre_processing(Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N)
    
    return Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N

if __name__ == '__main__':
    Ref_BC, Ref_O3, Ref_NO2, Ref_NO, Ref_PM10, Ref_N, Ref_Meteo, LCS_O3, LCS_NO2, LCS_NO, LCS_Meteo, LCS_PM1,LCS_PM25,LCS_PM10,LCS_N = main()
