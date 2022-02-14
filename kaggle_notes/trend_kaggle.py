# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 20:38:51 2022

@author: 12105
"""
import pandas as pd
import numpy as np

# trend: persistent, long-term change in the mean of the series
# moving average: average within a sliding time window

# linear trend:
target = a * time + b
# quadratic trend
target = a * time ** 2 + b * time + c

## moving average plot for trend in time series
# window of 365 days with the rolling method
moving_average = tunnel.rolling(
    window = 365,      # 365 day window
    center=True,       #puts the average at the center of the window
    min_periods=183,   #choose about hald the window size
).mean()               # compute the mean (or other aggregate function)

ax = tunnel.plot(style=".", color="0.5")
moving_average.plot(
    ax=asx, linewidth=3, title="Tunnel Traffic - 365-Day moving Average",legend=False,);

# for timeseries data, use the statsmodels library's DeterministicProcess function
# order is polynomial order, 1:linear, 2:quadratic, 3:cubic etc
from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
    index=tunnel.index,      # dates from the training data
    constant=True,           # dummy feature for the bias (y_intercept)
    order=1,                 # time dummy (trend)
    drop=True,               # drop terms if necessary to avoid collinearity
)
X = dp.in_sample()
X.head()

#	const	trend
#Day		
#2003-11-01	1.0	1.0
#2003-11-02	1.0	2.0
#2003-11-03	1.0	3.0
#2003-11-04	1.0	4.0
#2003-11-05	1.0	5.0

from sklearn.linear_model import LinearRegression

y = tunnel["NumVehicles"]  # the target

# The intercept is the same as the 'const' feature from 
# DeterministicProcess. LinearRegression behaves badly with duplicated
# features, so we need to be sure to exclude it here
model = LinearRegression(fit_intercept=False)
model.fit(X,y)

y_pred = pd.Series(model.predict(X), index=X.index)

# to make a forecast, apply the model to "out of sample" feautres
# this is times outside of the observation period of the training data
X = dp.out_of_sample(steps=30)
y_fore = pd.Series(model.predict(X), index=X.index)
y_fore.head()

#2005-11-17    114981.801146
#2005-11-18    115004.298595
#2005-11-19    115026.796045
#2005-11-20    115049.293494
#2005-11-21    115071.790944
#Freq: D, dtype: float64


 