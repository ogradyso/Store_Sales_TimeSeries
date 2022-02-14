#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 20:09:21 2022

@author: shauno
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

retail_sales = pd.read_csv(
    "./input/us-retail-sales.csv",
    parse_dates=['Month'],
    index_col='Month',
).to_period('D')
food_sales = retail_sales.loc[:, 'FoodAndBeverage']
auto_sales = retail_sales.loc[:, 'Automobiles']

dtype = {
    'store_nbr': 'category',
    'family': 'category',
    'sales': 'float32',
    'onpromotion': 'uint64',
}
store_sales = pd.read_csv(
    './input/train.csv',
    dtype=dtype,
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales = store_sales.set_index('date').to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family'], append=True)
average_sales = store_sales.groupby('date').mean()['sales']

###########################################
# 1: Determine trend with a moving average plot:
###########################################
ax = food_sales.plot()
ax.set(title="US Food and Beverage Sales", ylabel="Millions of Dollars");

# compute a moving
# average with appropriate parameters for trend estimation.

trend = food_sales.rolling(
    window=12,
    center=True,
    min_periods=6,
).mean()

# Make a plot
ax = food_sales.plot(alpha=0.5)
ax = trend.plot(ax=ax, linewidth=3)

##########################################
# 2: Identify trend
##########################################
# the trend fo the food and beverages data is quadratic

trend = average_sales.rolling(
    window=365,
    center=True,
    min_periods=183,
).mean()

ax = average_sales.plot(alpha=0.5)
ax = trend.plot(ax=ax, linewidth=3)

#########################################
# 3: Create a trend feature
#########################################
#Use the DeterministicProcess to create a feature set for a cubic trend model
#forecast for 90 days
from statsmodels.tsa.deterministic import DeterministicProcess

y = average_sales.copy() # the target

dp = dp = dp = DeterministicProcess(
    index=y.index,         # dummy feature for the bias (y_intercept)
    order=3,              # drop terms if necessary to avoid collinearity
)
X = dp.in_sample()

X_fore = dp.out_of_sample(steps=90)

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, linewidth=3, label="Trend", color='C0')
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color='C3')
ax.legend();

from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(index=y.index, order=11)
X = dp.in_sample()

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

ax = y.plot(alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, linewidth=3, label="Trend", color='C0')
ax.legend();

#########################################
# 4: Understand risks of forecasting with high-order polynomials
#########################################
# high-order polynomials are generally not well-suited to forecasting 

X_fore = dp.out_of_sample(steps=90)
y_fore = pd.Series(model.predict(X_fore), index=X_fore.index)

ax = y.plot(alpha=0.5, title="Average Sales", ylabel="items sold")
ax = y_pred.plot(ax=ax, linewidth=3, label="Trend", color='C0')
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color='C3')
ax.legend();

#########################################
# 5: Fit trend with splines
#########################################
# splines are an alternative to polynomials for fitting trends
#Multivariate Adaptive Regression Splines (MARS) from pyearth library

#from pyearth import Earth

# Target and features are the same as before
#y = average_sales.copy()
#dp = DeterministicProcess(index=y.index, order=1)
#X = dp.in_sample()

# Fit a MARS model with `Earth`
#model = Earth()
#model.fit(X, y)

#y_pred = pd.Series(model.predict(X), index=X.index)

#ax = y.plot(title="Average Sales", ylabel="items sold")
#ax = y_pred.plot(ax=ax, linewidth=3, label="Trend")

# with historical data, splines can be used to isolate other patterns
# in a time series by detrending:
#y_detrended = y - y_pred   # remove the trend from store_sales

#y_detrended.plot(title="Detrended Average Sales");
