#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 00:34:13 2022

@author: lisakov
"""

import sys 

import warnings
warnings.simplefilter(action='ignore')

from tqdm import tqdm

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.sarimax import SARIMAX
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.compose import make_reduction
#from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.ets import AutoETS
#from sktime.forecasting.bats import BATS
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.structural import UnobservedComponents
from sktime.forecasting.statsforecast import StatsForecastAutoARIMA

from sktime.utils.plotting import plot_series
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

from eda_tula import read_data

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
class Serie:
    """Common operation and data handling for time series.
    """
    
    def __init__(self, data, fh, metric=mean_absolute_percentage_error):
       self.data = data
       self.is_fitted = False
       self.metric_func = mean_absolute_percentage_error
       self.metric_kwargs = {'symmetric':False}
       self.metric_value = np.inf
       self.best_algo = 'naive'
       self.fh = fh
       self.evals = pd.DataFrame(columns=['algo', 'fh_length', 'metric_value']) # used to choose the best algo. 
#        real prediction is done once 
       self.predictions = np.array([])
       self.best_prediction = 0
       self.best_prediction_idx = 0
       
    def evaluate(self, y_true, y_pred):
        self.metric_value = self.metric_func(y_true, y_pred, **self.metric_kwargs)
        return self.metric_value
        
    def run_prophet(self):
       pass
   
    def run_autoarima(self, y_test):
        
        if (~data.isna()).sum().values[0] < 12:
            if (~data.isna()).sum().values[0] / data.index.size < 0.1:
                algo = 'null'
                self.y_pred = data.iloc[-1,:] * 0
            else:
                algo = 'last'
                self.y_pred = data.iloc[-1,:]
        else:
            forecaster = AutoARIMA(sp=12)
            algo = 'autoarima'
            try:
                forecaster.fit(data)
                self.y_pred = forecaster.predict(self.fh)
                self.y_pred[self.y_pred < 0 ] = 0  # in case of errors
            except:
                algo = 'last2'
                self.y_pred = data.iloc[-1,:]


        self.y_pred.index = y_test.index
        self.evals = pd.concat([self.evals, 
                                pd.DataFrame(columns=self.evals.columns, 
                                data=[[algo, len(fh), self.evaluate(y_test, self.y_pred)]])])
            
        self.predictions = np.append(self.predictions.flatten(), self.y_pred).reshape(-1, len(fh))
            
        
        
        
        self.is_fitted = True
        
    def report(self):
#            best_algo = self.evals
#        print('Best algo is:')
        print(self.evals)




################# mAIN
    
train_file = '/home/lisakov/Programs/chempIIonat2022/tula/train_dataset_train.csv'
test_file = '/home/lisakov/Programs/chempIIonat2022/tula/test_dataset_test.csv'



cols_x = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'VISIT_MONTH_YEAR', 'AGE_CATEGORY']
cols_f = ['PATIENT_SEX', 'MKB_CODE', 'ADRES', 'AGE_CATEGORY']
cols_t = ['VISIT_MONTH_YEAR', 'PATIENT_ID_COUNT']
cols_y = ['PATIENT_ID_COUNT']

# train file
#df = read_data(train_file)
dt = read_data(test_file)
DT = read_data(test_file)
DT.loc[:, 'PATIENT_ID_COUNT'] = 1  # in case a combination of features did no show up in a training dataset:
# the decease is rare (was always zero in training)
# it appeared in the test, hence it is not zero
# rare+nonzero = 1


# load modified dataset 
with open('train_mod.pkl', 'rb') as f:
    df = pkl.load(f)



visit_dates= np.sort(df.VISIT_MONTH_YEAR.unique())
idx = pd.PeriodIndex(data=visit_dates, freq='1m')

n_predict = 1
rarity_cutoff = 10 # do some real fitting only on the data with more than this amount of patients
TEST_METRIC = True # to test performance of different metrics. 

df.reset_index(inplace=True) # save index in a column
#df2fit = df.loc[df.total_n_patients > rarity_cutoff].groupby(cols_f)['index'].agg(list)
df2fit = df.loc[(df.total_n_patients > 10) & (df.total_n_patients < 12)].groupby(cols_f)['index'].agg(list)

metrics = np.zeros(df2fit.index.size)
best_algo = np.zeros(df2fit.index.size).astype(str) # the best performing algorithm 
use_prophet, use_autoarima, use_tbats, use_naive = 0,1,0,0

results = pd.DataFrame(columns=['prophet', 'autoarima', 'tbats', 'naive', 'best_metric', 'best_algo'], index=range(df2fit.index.size))



for i in tqdm(range(df2fit.index.size)):
    data2 = pd.DataFrame(index=idx)
    data = df.iloc[df2fit.iloc[i]][cols_t]
    
    
    
    data = data.sort_values(by='VISIT_MONTH_YEAR')
    data.VISIT_MONTH_YEAR = data.VISIT_MONTH_YEAR.dt.to_period('M')
    data.set_index('VISIT_MONTH_YEAR', inplace=True)
    data.index.name = None

#    data = data2.merge(data, how='left', left_index=True, right_on='VISIT_MONTH_YEAR')
    data = data2.join(data, how='left').fillna(0.)

    print(data)
    
    if TEST_METRIC is True:
        y_train, y_test = temporal_train_test_split(data, test_size=n_predict) 
    # define the horizon of forecasting
    fh = ForecastingHorizon(y_test.index, is_relative=False)

#     deal with vaccination which started only in the second half of the series
#    if y_train.iloc[:int(y_train.index.size/2)].mean().values[0] * 10 < y_train.iloc[int(y_train.index.size/2):].mean().values[0]:
#        y_train = y_train.iloc[int(y_train.index.size/2):]
    
    met = {'prophet': np.inf, 'autoarima': np.inf, 'tbats': np.inf, 'naive': np.inf }
    
    if use_prophet:
        # prophet does not support periodIndex
        z_train = y_train.copy()
        z_train.index = z_train.index.to_timestamp(freq='M')
        z_test = y_test.copy()
        z_test.index = z_test.index.to_timestamp(freq='M')
    
        forecaster = Prophet(
            freq='M',
            seasonality_mode="additive",
            yearly_seasonality=20,
            uncertainty_samples=False
        )

        forecaster.fit(z_train)
        y_pred = forecaster.predict(fh.to_relative(cutoff=y_train.index[-1]))
        y_pred[y_pred < 0 ] = 0  # in case of errors
        y_pred.index = y_test.index
        met['prophet'] = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
        results.iloc[i]['prophet'] = met['prophet']

    
    if use_autoarima:
        s = Serie(data=y_train, fh=fh)
        s.run_autoarima(y_test)
        s.report()
        
#        forecaster = AutoARIMA(sp=12)
#        try:
#            forecaster.fit(y_train)
#            y_pred = forecaster.predict(fh)
#            y_pred[y_pred < 0 ] = 0  # in case of errors
#            y_pred.index = y_test.index
#            met['autoarima'] = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
#            results.iloc[i]['autoarima'] = met['autoarima']
#        except:
#            pass

    if use_tbats:
        forecaster = TBATS(sp=12, use_trend=True, use_box_cox=False)
        try:
            forecaster.fit(y_train)
            y_pred = forecaster.predict(fh)
            y_pred[y_pred < 0 ] = 0  # in case of errors
            y_pred.index = y_test.index
            met['tbats'] = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
            results.iloc[i]['tbats'] = met['tbats']
        except:
            pass
        
    if use_naive:
        forecaster = NaiveForecaster(strategy='last', sp=12)
        try:
            forecaster.fit(y_train)
            y_pred = forecaster.predict(fh)
            y_pred[y_pred < 0 ] = 0  # in case of errors
            y_pred.index = y_test.index
            met['naive'] = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
            results.iloc[i]['naive'] = met['naive']
        except:
            pass

    
    best_algo[i] = min(met, key=met.get)
    metrics[i] = met[best_algo[i]]
    
    results.iloc[i]['best_metric'] = results.iloc[i].min()
    results.iloc[i]['best_algo'] = min(met, key=met.get)
    
    # if all normal algorithms failed, fill predicition with 
    if results.iloc[i]['best_metric'] is np.nan:
        y_pred = np.ceil(y_test[~y_test.isna()].mean())
        results.iloc[i]['best_metric'] = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
        results.iloc[i]['best_algo'] = 'aver'

    if results.iloc[i]['best_metric'] is np.nan:
        y_pred.iloc[0] = 1.0
        results.iloc[i]['best_metric'] = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
        results.iloc[i]['best_algo'] = 'unity'


#    plot_series(data, y_pred, labels=['data', 'pred'])
#    metrics[i] = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
    


print('\n\nAverage metric is {:.3f}\n\n\n'.format(results['best_metric'].mean()))
print('Algos provided the best metric:\n{}'.format(results.groupby('best_algo')['best_metric'].count() / results.index.size))
sns.displot(data=results, kind='kde')

sys.exit(55)




# try prediction on a well populated time series
n_show = 1
data = df.loc[df.total_n_patients == df.total_n_patients.nlargest(200).values[51], ['VISIT_MONTH_YEAR', 'PATIENT_ID_COUNT']]

# prev version
data.VISIT_MONTH_YEAR = data.VISIT_MONTH_YEAR.dt.to_period('M')
data.set_index('VISIT_MONTH_YEAR', inplace=True)
#                print(data)
#
dd = pd.DataFrame(index=idx, data=np.zeros(idx.size), columns=['PATIENT_ID_COUNT'] )
dd.loc[data.index, 'PATIENT_ID_COUNT'] = data.PATIENT_ID_COUNT


y_train, y_test = temporal_train_test_split(dd, test_size=1) # there are 51 measurements

# define the horizon of forecasting
fh = ForecastingHorizon(y_test.index, is_relative=False)

# test different forecasters
forecaster0 = NaiveForecaster(strategy='last', sp=12)
forecaster1 = SARIMAX(order=(1, 0, 1), trend=[1,0], seasonal_order=(0, 0, 1, 12))
forecaster2 = AutoARIMA(sp=12)
#forecaster3 = ExponentialSmoothing(trend="add", seasonal="additive", sp=6) # -
#forecaster3 = AutoETS(auto=True, sp=12, n_jobs=-1)
forecaster3 = TBATS(sp=12, use_trend=True, use_box_cox=False)
#

forecaster3 = StatsForecastAutoARIMA(sp=12)
#
forecaster3 = Prophet(
    seasonality_mode="multiplicative",
    n_changepoints=int(len(y_train) / 12),
    add_country_holidays={"country_name": "Russia"},
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
)

#forecaster3 = UnobservedComponents(
#    level="local linear trend", freq_seasonal=[{"period": 12, "harmonics": 3}]
#)


#
#forecaster3.fit(z_train)
#y_pred = forecaster3.predict(fh.to_relative(cutoff=y_train.index[-1]))
#y_pred.index = y_test.index


forecasters = [forecaster0, forecaster1, forecaster2, forecaster3]


#forecasters = [forecaster3]

#z = dd.copy()
#z = z.to_timestamp(freq="M")
#z_train, z_test = temporal_train_test_split(z, test_size=15)
#

#forecasters = [ARIMA(order=(1, 1, 0), seasonal_order=(0, 1, 0, 12), suppress_warnings=True) for i in [0,1,2,3]]
#forecasters = [SARIMAX(order=(1, 0, 1), seasonal_order=(0, 0, 1, 12), trend=[1,0]) for i in [0,1]]

#forecasters = [ARIMA(order=(1, i, 0), seasonal_order=(1, 1, 1, 12), suppress_warnings=True) for i in [0,1,2,3]]




y_preds = dict.fromkeys(list(range(len(forecasters))))
labels = [str(fo) for fo in forecasters]
metrics = np.zeros(len(forecasters))



for i,f in enumerate(forecasters):
    f.fit(y_train)
    y_pred = f.predict(fh)
#    print(y_pred)
    y_preds[i] = y_pred
    metrics[i] = mean_absolute_percentage_error(y_test, y_pred, symmetric=False)
#    print('{}. mape = {:.2f} '.format(f, metrics[i]))
    



plot_series(dd,  *y_preds.values(), labels=['data'] + labels)
for i,j in zip(labels, metrics):
    print('{}: {:.2f}'.format(i,j))

#sns.relplot(x=labels, y=metrics)





##DT.loc[(DT.MKB_CODE == m) & (DT.ADRES == a) & (DT.AGE_CATEGORY == ag) & (DT.PATIENT_SEX == p), 'PATIENT_ID_COUNT'] = np.ceil(y_pred.values)
#
#
#
#plot_series(dd)



#
#sys.exit(57)
#
##mkb_codes = df.MKB_CODE.unique()
##adreses = df.ADRES.unique()
##patient_sexes = df.PATIENT_SEX.unique()
##age_categories = df.AGE_CATEGORY.unique()
##visit_dates= np.sort(df.VISIT_MONTH_YEAR.unique())
#
#
#dt_mkb_codes = dt.MKB_CODE.unique()
#
#
#pdate = DT.VISIT_MONTH_YEAR.unique()
#idx = pd.PeriodIndex(data=visit_dates, freq='1m')
#
#for m in dt_mkb_codes:
#    print('MKB_CODE = {}'.format(m))
#    dtm = dt.loc[dt.MKB_CODE == m,:]
#    dt_adreses = dtm['ADRES'].unique()
#    print('adreses = {}'.format(dt_adreses))
#    for a in dt_adreses:
#        print('  ADRES = {}'.format(a))
#        dtma = dtm.loc[dtm.ADRES == a]
#        dt_age_categories = dtma.AGE_CATEGORY.unique()
#        print('  age categories = {}'.format(dt_age_categories))
#        for ag in dt_age_categories:
#            print('    AGE_CATEGORY = {}'.format(ag))
#            dtmaa = dtma.loc[dtma.AGE_CATEGORY == ag]
#            dt_patient_sexes = dtmaa.PATIENT_SEX.unique()
#            print('    sexes = {}'.format(dt_patient_sexes))
#            for p in dt_patient_sexes:
#                print('      PATIENT_SEX = {}'.format(p))
#                dtmaap = dtmaa.loc[dtmaa.PATIENT_SEX == p]
#                
#                data = df.loc[(df.MKB_CODE == m) & (df.ADRES == a) & (df.AGE_CATEGORY == ag) & (df.PATIENT_SEX == p), ['VISIT_MONTH_YEAR', 'PATIENT_ID_COUNT']]
#                data.VISIT_MONTH_YEAR = data.VISIT_MONTH_YEAR.dt.to_period('M')
#                data.set_index('VISIT_MONTH_YEAR', inplace=True)
##                print(data)
#                
#                dd = pd.DataFrame(index=idx, data=np.zeros(idx.size), columns=['PATIENT_ID_COUNT'] )
#                dd.loc[data.index, 'PATIENT_ID_COUNT'] = data.PATIENT_ID_COUNT
#                
#                forecaster = SARIMAX(order=(1, 1, 1), trend="t", seasonal_order=(0, 0, 0, 0))
#                forecaster.fit(dd)
#                y_pred = forecaster.predict(fh=1)
#                DT.loc[(DT.MKB_CODE == m) & (DT.ADRES == a) & (DT.AGE_CATEGORY == ag) & (DT.PATIENT_SEX == p), 'PATIENT_ID_COUNT'] = np.ceil(y_pred.values)
#
#DT.loc[DT.PATIENT_ID_COUNT <= 0, 'PATIENT_ID_COUNT'] = 1 # correct unphysical predictions
#
#DT.PATIENT_ID_COUNT = DT.PATIENT_ID_COUNT.astype(int)
#DT.loc[:, 'month'] = DT.VISIT_MONTH_YEAR.dt.month.astype(str).str.zfill(2)
#DT.loc[:, 'year'] = (DT.VISIT_MONTH_YEAR.dt.year - 2000).astype(str).str.zfill(2)
#DT.VISIT_MONTH_YEAR = DT.month + '.' + DT.year
#DT.drop(['month', 'year'], axis=1, inplace=True)
#
#DT.to_csv('/home/lisakov/Programs/chempIIonat2022/tula/prediction2.csv', sep=';', index=False)
