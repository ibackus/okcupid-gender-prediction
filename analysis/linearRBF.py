#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:03:37 2016

@author: ibackus
"""
import numpy as np

import RBFkernel
import sgd
import utils
import fitutils

class LinearRBF():
    
    def __init__(self, x, y, scalarOffset=True, learningRateScale=1e-6, 
                 regScale=0.0, miniBatchSize=30, arrayBatchSize=700,
                 lossScheme='linear binary', tau0=1., kappa=0.):
        """
        Trains a linear model using the RBFKernel as a feature map, using
        gradient descent.  Assumes 2 classes only.  Available loss schemes are:
            
            'softmax'
            'linear binary'
        """
        # Load data into self
        self.x = x
        self.y = y
        self.results = {}
        self.regScale = regScale
        self.miniBatchSize = miniBatchSize
        self.arrayBatchSize = arrayBatchSize
        self.tau0 = tau0
        self.kappa = kappa
        # Perform initialization setup
        self._setup(scalarOffset, learningRateScale, lossScheme)
        
    def _setup(self, scalarOffset=None, learningRateScale=None, lossScheme=None):
        """
        Setup functions to use (feature maps, etc)
        """
        self.initFeatureMap(scalarOffset)
        self.initLearningRate(learningRateScale)
        self.initSGD(lossScheme)
            
    def __getstate__(self):
        
        ignore = ('sgd', 'featureMap', 'learningRate')
        result = self.__dict__.copy()
        for key in ignore:
            
            if key in result:
                
                result.pop(key, None)
        
        return result
        
    def __setstate__(self, dictionary):
        
        self.__dict__ = dictionary
        self._setup()
        self.sgd._initdata(self.x, self.y, self.regScale, self.miniBatchSize)
    
    def initFeatureMap(self, scalarOffset=None):
        """
        
        """
        # Parse args/setup defaults
        
        if scalarOffset is None:
            scalarOffset = self.scalarOffset
        self.scalarOffset = scalarOffset
        # Set up feature map
        self.featureMap = RBFkernel.RBFkernel(self.x, scalarOffset=scalarOffset)
        
    
    def initLearningRate(self, learningRateScale=None):
        """
        """
        if learningRateScale is None:
            learningRateScale = self.learningRateScale
        self.learningRateScale = learningRateScale
        np.random.seed(0)
        ind = np.random.rand(len(self.x)).argsort()[0:100]
        h = self.featureMap(self.x[ind])
        self.learningRate = fitutils.powerlawLearningRate(h, \
            self.learningRateScale, self.kappa, self.tau0)
        
    def initSGD(self, lossScheme=None):
        """
        Initialize the SGD instance which will be used to perform the fit
        
        """
        if lossScheme is None:
            lossScheme = self.lossScheme
        self.lossScheme = lossScheme
        self.sgd = sgd.SGD(self.learningRate, miniBatchSize=self.miniBatchSize, \
            featureMapFcn=self.featureMap, lamScale=self.regScale, \
            arrayBatchSize=self.arrayBatchSize, lossScheme=lossScheme)
        
    def subset(self, N=1000):
        """
        Generate a random subset of data points from self.x and self.y
        """
        ind = np.random.rand(len(self.x)).argsort()[0:N]        
        x2 = self.x[ind]
        y2 = self.y[ind]
        return x2, y2
    
    def fit(self, **kwargs):
        """
        kwargs are passed to sgd.SGD.fit()
        
        Additional kwargs
        ------------------
                    
        subset: int
            Use this many data points to test
        """
        # Handle defaults
        self.regScale = kwargs.get('lamScale', self.regScale)
        kwargs['lamScale'] = self.regScale
        self.miniBatchSize = kwargs.get('miniBatchSize', self.miniBatchSize)
        kwargs['miniBatchSize'] = self.miniBatchSize
        # Save to self
        self._fitkwargs = kwargs
        x = self.x
        
        if 'subset' in kwargs:
            N = kwargs['subset']
            kwargs.pop('subset', None)
            x, y = self.subset(N)
        else:
            y = self.y
        # Do the fit and store results
        try:
            self.sgd.fit(x, y, **kwargs)
        finally:
            self.results = self.sgd.results()
            
    def pred(self, x):
        """
        Pred class of x
        """
        if (self.arrayBatchSize is not None) and (len(x) > self.arrayBatchSize):
            # Calculate y_predicted in batches
            ypred = np.zeros(len(x))
            for slicer in utils.arrayBatch(x, self.arrayBatchSize):
                
                ypred[slicer] = self.pred(x[slicer]).flatten()
        
        else:
                
            h = self.featureMap(x)
            ypred = self.sgd.classifier(h, self.results['w'])
        
        return utils.columnVector(ypred)

        
    def refit(self, **kwargs):
        """
        Re-do the previous fit, using the previous weights as a starting
        point.  Additional kwargs can be passed to the fit function
        sgd.SGD.fit
        """
        w = self.results.get('w', None)
        arguments = self._fitkwargs
        arguments.update(kwargs)
        arguments['w'] = w
        self.fit(self.y, **arguments)