#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 16:07:56 2016

@author: ibackus
"""

import numpy as np
import cPickle as pickle
import okcupidio
import pandas as pd

# RUN FLAGS
do_make_features = True
do_filter_features = True
do_pca = True

# SETTINGS
savename = 'features.p'
minResponseRate = 0.65
nTest = 2000

# CODE
if do_make_features:
    # Loads data (fairly raw)
    data = okcupidio.loadData()
    # Preprocess data
    okcupidio.processData(data)
    # And make binary features out of it
    features, featureNames = okcupidio.buildFeatures(data)
    # First, replace -1 with nans
    features[features == -1] = np.nan
    # Save
    pickle.dump((features, featureNames), open(savename, 'w'), 2)
else:
    features, featureNames = pickle.load(open(savename, 'r'))

if do_filter_features:
    
    df = pd.DataFrame(data=features, columns=featureNames)
    columns = df.columns[df.columns != 'income']
    # Ignore data with response rates too low
    for col in columns:
        
        if df[col].notnull().mean() < minResponseRate:
            
            df.drop(col, 1, inplace=True)
            
    # Now replace missing data (EXCEPT IN INCOME)
    columns = df.columns[df.columns != 'income']
    for col in columns:
        series = df[col]
        nChoice = series.isnull().sum()
        if nChoice > 0:
            notnulls = series[series.notnull()]
            vals = np.random.choice(notnulls, nChoice)
            series[series.isnull()] = vals
            df[col] = series
    
    # Now remove people without reported incomes
    df = df[df.income.notnull()]
    # Ignore entries where stddev == 0
    columns = df.columns[df.columns != 'income']
    for col in columns:
        if df[col].std() == 0:
            df.drop(col, 1, inplace=True)
            
    # drop last_online
    df.drop('last_online', 1, inplace=True)
    # Give us arrays to work with
    income = df.income
    df = df.drop('income', 1)
    names = df.columns.tolist()
    x = df.values
    y = income.values
    
    pickle.dump((x, y, names), open('dataset.p', 'w'), 2)
    
else:
    
    x, y, names = pickle.load(open('dataset.p', 'r'))
    
# NOW SPLIT the dataset
y = y.reshape([len(y), 1])
nTrain = len(x) - nTest
np.random.seed(0)
ind = np.random.rand(len(x)).argsort()
trainInd = ind[0:nTrain]
testInd = ind[nTrain:]
xtrain, ytrain = (x[trainInd], y[trainInd])
xtest, ytest = (x[testInd], y[testInd])
# save the split set
pickle.dump((xtrain, ytrain), open('data_train.p', 'w'), 2)
pickle.dump((xtest, ytest), open('data_test.p','w'), 2)
