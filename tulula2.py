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
from sktime.forecasting.sarimax import SARIMAX



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
        df.loc[:, 'PATIENT_ID_COUNT'] = df.loc[:, 'PATIENT_ID_COUNT'].astype(float)
    except:
        pass
    
    return df



    
    
  


################# mAIN
    
train_file = '/home/lisakov/Programs/chempIIonat2022/tula/train_dataset_train.csv'
test_file = '/home/lisakov/Programs/chempIIonat2022/tula/test_dataset_test.csv'



cols_x = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY']
cols_y = ['PATIENT_ID_COUNT']

# train file
df = read_data(train_file)
dt = read_data(test_file)
DT = read_data(test_file)
DT.loc[:, 'PATIENT_ID_COUNT'] = 1  # in case a combination of features did no show up in a training dataset:
# the decease is rare (was always zero in training)
# it appeared in the test, hence it is not zero
# rare+nonzero = 1

#mkb_codes = df.MKB_CODE.unique()
#adreses = df.ADRES.unique()
#patient_sexes = df.PATIENT_SEX.unique()
#age_categories = df.AGE_CATEGORY.unique()
#visit_dates= np.sort(df.VISIT_MONTH_YEAR.unique())


dt_mkb_codes = dt.MKB_CODE.unique()


pdate = DT.VISIT_MONTH_YEAR.unique()
idx = pd.PeriodIndex(data=visit_dates, freq='1m')

for m in dt_mkb_codes:
    print('MKB_CODE = {}'.format(m))
    dtm = dt.loc[dt.MKB_CODE == m,:]
    dt_adreses = dtm['ADRES'].unique()
    print('adreses = {}'.format(dt_adreses))
    for a in dt_adreses:
        print('  ADRES = {}'.format(a))
        dtma = dtm.loc[dtm.ADRES == a]
        dt_age_categories = dtma.AGE_CATEGORY.unique()
        print('  age categories = {}'.format(dt_age_categories))
        for ag in dt_age_categories:
            print('    AGE_CATEGORY = {}'.format(ag))
            dtmaa = dtma.loc[dtma.AGE_CATEGORY == ag]
            dt_patient_sexes = dtmaa.PATIENT_SEX.unique()
            print('    sexes = {}'.format(dt_patient_sexes))
            for p in dt_patient_sexes:
                print('      PATIENT_SEX = {}'.format(p))
                dtmaap = dtmaa.loc[dtmaa.PATIENT_SEX == p]
                
                data = df.loc[(df.MKB_CODE == m) & (df.ADRES == a) & (df.AGE_CATEGORY == ag) & (df.PATIENT_SEX == p), ['VISIT_MONTH_YEAR', 'PATIENT_ID_COUNT']]
                data.VISIT_MONTH_YEAR = data.VISIT_MONTH_YEAR.dt.to_period('M')
                data.set_index('VISIT_MONTH_YEAR', inplace=True)
#                print(data)
                
                dd = pd.DataFrame(index=idx, data=np.zeros(idx.size), columns=['PATIENT_ID_COUNT'] )
                dd.loc[data.index, 'PATIENT_ID_COUNT'] = data.PATIENT_ID_COUNT
                
                forecaster = SARIMAX(order=(1, 1, 1), trend="t", seasonal_order=(0, 0, 0, 0))
                forecaster.fit(dd)
                y_pred = forecaster.predict(fh=1)
                DT.loc[(DT.MKB_CODE == m) & (DT.ADRES == a) & (DT.AGE_CATEGORY == ag) & (DT.PATIENT_SEX == p), 'PATIENT_ID_COUNT'] = np.ceil(y_pred.values)

DT.loc[DT.PATIENT_ID_COUNT <= 0, 'PATIENT_ID_COUNT'] = 1 # correct unphysical predictions

DT.PATIENT_ID_COUNT = DT.PATIENT_ID_COUNT.astype(int)
DT.loc[:, 'month'] = DT.VISIT_MONTH_YEAR.dt.month.astype(str).str.zfill(2)
DT.loc[:, 'year'] = (DT.VISIT_MONTH_YEAR.dt.year - 2000).astype(str).str.zfill(2)
DT.VISIT_MONTH_YEAR = DT.month + '.' + DT.year
DT.drop(['month', 'year'], axis=1, inplace=True)

DT.to_csv('/home/lisakov/Programs/chempIIonat2022/tula/prediction2.csv', sep=';', index=False)
