# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 20:38:51 2022

@author: 12105
"""
import pandas as pd
import numpy as np

# Time series, lesson 1:
df = pd.read_csv('input_data.csv')
# Two types of features: time-step and lag features

### Time-step features can be derived directly from the time index
#Most basic time-step feature: time dummy (count of time steps in series from beginning to end)

df['Time'] = np.arange(len(df.index))

# Lag features

df['Lag_1'] = df['Hardcover'].shift(1)
df = df.reindex(columns=['Hardcover','Lag_1'])

# observations are shifted in time (similar to lag/lead window functions in SQL)
# allows for investigating the effect of the previous observations 

# Lag plots view observations plotted against the previous observation

#Lag features allow the modeling of serial dependence:
# data can be predicted from previous observations

#pandas' shift method is used to lag a series
df['Lag_1'] = df['NumVehicles'].shift(1)

# DROP missing values and the corresponding targets:
from sklearn.linear_model import LinearRegression

X = df.loc[:, ['Lag_1']]
X.dropna(inplace=True)
y = df.loc[:, 'NumVehicles']
y, X = y.align(X, join='inner')

model = LinearRegression()
model.fit(X,y)

y_pred = pd.Series(model.predict(X), index=X.index)

