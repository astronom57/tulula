#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 19:11:10 2022

@author: lisakov
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 00:34:13 2022

@author: lisakov
"""

import sys 

import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

import pickle as pkl

def read_data(file):
    """Read data from either train or test data files.
    
    Args:
        file (str): filename to read
        
    Returns:
        df (dataframe): data in the form of a pandas dataframe
    """
    
    df = pd.read_csv(file, sep=';', dtype=str)
    
    df.loc[:, 'VISIT_MONTH_YEAR'] = pd.to_datetime(df.loc[:, 'VISIT_MONTH_YEAR'], format='%m.%y')
    df.loc[:, 'PATIENT_SEX'] = df.loc[:, 'PATIENT_SEX'].astype(int)
    try:
        df.loc[:, 'PATIENT_ID_COUNT'] = df.loc[:, 'PATIENT_ID_COUNT'].astype(int)
    except:
        pass
    
    return df



def show_frequent(df, n: int):
    """Plot n most frequent categories of each feature.
    
    Args:
        df: pandas dataframe 
        n: number of categories to plot. If a feature has less categories tahn n, plot all of them.
        
    Returns:
        nothing
    """
    
    for col in df.loc[:, df.columns != 'PATIENT_ID_COUNT'].columns:
        temp = df.groupby(col)[['PATIENT_ID_COUNT']].sum().nlargest(n_show, 'PATIENT_ID_COUNT')
        temp.reset_index(inplace=True)
        plot = sns.catplot(data=temp, x=col, y='PATIENT_ID_COUNT', kind='bar')
        plot.ax.set_title(col)
    return


################# MAIN
if __name__ == '__main__':
        
    train_file = '/home/lisakov/Programs/chempIIonat2022/tula/train_dataset_train.csv'
    test_file = '/home/lisakov/Programs/chempIIonat2022/tula/test_dataset_test.csv'
    
    
    cols_x = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY']
    cols_f = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY']
    cols_y = ['PATIENT_ID_COUNT']
    
    # train file
    df = read_data(train_file)
    dt = read_data(test_file)
    
    
    # inspect data cardinality
    print('Number of categories per variable:\n{}'.format(df.apply(pd.Series.nunique)))
    
    
    # take a look at the most frequent deceases
    n_show = 20
    #show_frequent(df, n_show)
    
    
    
    # plot time series for selected categories
    n_show = 2
    # most frerquenct deceases in big cities
    for city in df.groupby('ADRES')[['PATIENT_ID_COUNT']].sum().nlargest(n_show, 'PATIENT_ID_COUNT').index:
        for mkb in df.groupby('MKB_CODE')[['PATIENT_ID_COUNT']].sum().nlargest(n_show, 'PATIENT_ID_COUNT').index:
            temp = df.query('ADRES == @city and MKB_CODE == @mkb')
    #        print(temp)
            
            plot = sns.relplot(data=temp, x='VISIT_MONTH_YEAR', y='PATIENT_ID_COUNT', 
                        hue='AGE_CATEGORY', col='PATIENT_SEX',
                        kind='line')
            plot.fig.suptitle(f'city={city}, mkb={mkb}')
            
    # least frequent deceases in small towns
    n_show = 1
    for city in df.groupby('ADRES')[['PATIENT_ID_COUNT']].sum().nsmallest(n_show, 'PATIENT_ID_COUNT').index:
        for mkb in df.loc[df.ADRES == city].groupby('MKB_CODE')[['PATIENT_ID_COUNT']].sum().nsmallest(n_show, 'PATIENT_ID_COUNT').index:
            temp = df.query('ADRES == @city and MKB_CODE == @mkb')
            print(temp)
            
            plot = sns.relplot(data=temp, x='VISIT_MONTH_YEAR', y='PATIENT_ID_COUNT', 
                        hue='AGE_CATEGORY', col='PATIENT_SEX',
                        kind='scatter')
            plot.fig.suptitle(f'city={city}, mkb={mkb}')
    
    # How many time series are dominated by zeroes. 
    # Using of a simplified pforecasting method will reduce training time. 
    #total_n_timeseries = np.prod([df[c].nunique() for c in cols_f]) # total possible number of feature combinations
    ts = df.groupby(cols_f).sum()
    total_n_timeseries =  ts.index.size # real number of timeseries
    # the minimal number of patients is one in ~four years. Count the ratio of such time series to the total number
    ratio_ones = ts[ts.PATIENT_ID_COUNT == 1].index.size / total_n_timeseries # 40%
    # count time series with number of patients less or equal to 1 per month
    n_months = df.VISIT_MONTH_YEAR.unique().size # 51
    ratio_rare = ts[ts.PATIENT_ID_COUNT < n_months].index.size / total_n_timeseries # 95%
    
    sns.displot(data=ts[ts.PATIENT_ID_COUNT < n_months], kind='hist')
    # To be 2 sigma certain about predicting zero patients, predict it only for time series
    # with approximately >= 95% of zeroes in the data. 48/51 = 94%. So prediction to time series 
    # with <= 3 patients over the course of n_months is zero.
    ratio_rare_95percent = ts[ts.PATIENT_ID_COUNT <= 3 ].index.size / total_n_timeseries # 65%
    # the only way I see to decide between 0 and 1 for rare cases prediction
    ratio_rare_50percent = ts[ts.PATIENT_ID_COUNT < n_months / 2 ].index.size / total_n_timeseries # 92% will be 0
    
    
    
    def transform_df(df, test=False):
        """Properly transform data in the dataframe for using in training.
        Dates -> datetimeindex
        add required columns
        """
        if test:
            df.loc[:, 'PATIENT_ID_COUNT'] = 1  # add filler values to the test data. 1 because 
        
        dp = df.pivot_table(values='PATIENT_ID_COUNT',columns='VISIT_MONTH_YEAR', index=cols_f, fill_value=np.nan)
        
        if not test:
            dp.loc[:, dp.columns.max() + pd.DateOffset(months=1)] = np.nan
        
        return dp
        
        
    df_pivot = transform_df(df)    
    dt_pivot = transform_df(dt, test=True)
    
    with open('train_pivot.pkl', 'wb') as f:
        pkl.dump(df_pivot, f)

    with open('test_pivot.pkl', 'wb') as f:
        pkl.dump(df_pivot, f)
        
     
    # to bring df to a lond format: 
    # reset_index()
    # pd.melt(id_vars=cols_f)
        
 
#    # prepare dataframe to use in training
#    df_mod = df.copy()
#    df_mod.loc[:, 'total_n_patients'] = df.groupby(cols_f)['PATIENT_ID_COUNT'].transform(sum)
#    
#    with open('train_mod.pkl', 'wb') as f:
#        pkl.dump(df_mod, f)
        
        
        
        
        
        
    