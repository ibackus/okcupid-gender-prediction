#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:48:22 2016

@author: ibackus
"""
import numpy as np
import cPickle as pickle
import utils
from linearBinaryClassifier import LinearBinaryClassifier

class BinaryLinearRegression():
    """
    Class for performing binary linear regression
    """
    def __init__(self, regScale=1e-1, classify=True, scalarOffset=True):
        
        self.regScale = regScale
        self.classify = classify
        self.scalarOffset = scalarOffset

    def fit(self, x, y, regScale=None, threshold=None, dataWeights=None):
        """
        Perform a linear (ridge) regression
        
        Parameters
        ----------
        x : array
            Training features, shape (n, d)
        y : array
            Labels.  shape (n, 1)
        regScale : float
            Amount to scale regularization by
        
        Saves
        -------
        w : weights
            Approximate solution to equation y = x.dot.w
        """
        # Initialize
        y = utils.columnVector(y)
        self._xraw = x
        if regScale is not None:
            
            self.regScale = regScale
            
        x, center, scale = utils.condition(x)
        self.x = x
        self._center = center
        self._scale = scale
        self.y = y
        
        
        # Apply weights to data points        
        if dataWeights is not None:
            
            dataWeights = utils.columnVector(dataWeights)
            sqrtWeights = np.sqrt(dataWeights)
            x = x * sqrtWeights
            y = y * sqrtWeights
            
        self.dataWeights = dataWeights
        self.regularization = self.regScale/(x**2).mean()
            
        # Perform the regression
        M = np.dot(x.T, x)
        ind = np.arange(len(M))
        M[ind, ind] += self.regularization
        xTy = np.dot(x.T, y)
        self.w = np.dot(np.linalg.inv(M), xTy)
        if np.any(np.isnan(self.w)):
            
            raise RuntimeError, "NaN encounter in w in fit"
            
        # Get scalar offset
        if self.scalarOffset:
            self.w0 = (y - np.dot(x, self.w)).mean()
        else:
            self.w0=0.
        if self.classify:
            # Set up the classifier (using the un-weighted data)
            self.classifier = LinearBinaryClassifier(self.y, self.ypred(self.x))
        
        
    def pred(self, x):
        """
        binary classification of x
        """
#        # Condition the data, using the same conditioning as for the training
#        # set
#        ypred = self.ypred(x)
#        classPred = np.zeros([len(ypred), 1])
#        classPred[ypred > self.threshold] = 1
#        return classPred
        return self.classifier.classify(self.ypred(x))
        
    def ypred(self, x):
        """
        x.dot.w
        """
        x, dummy1, dummy2 = utils.condition(x, self._center, self._scale)
        ypred = np.dot(x, self.w) + self.w0
        return ypred
        
    def confidence(self, x):
        """
        Return the confidence of a classification
        """
        return self.classifier.confidence(self.ypred(x))

class LinearRegression():
    """
    Class for performing mutli-class linear regression
    """
    def __init__(self, regScale=1e-1):
        
        self.regScale = regScale

    def fit(self, x, y, regScale=None):
        """
        Perform a linear (ridge) regression
        
        Parameters
        ----------
        x : array
            Training features, shape (n, d)
        y : array
            Labels.  shape (n, nClass)
        regScale : float
            Amount to scale regularization by
        
        Saves
        -------
        w : weights
            Best solution to equation y = x.dot.w
        """
        if regScale is not None:
            
            self.regScale = regScale
            
        x, center, scale = utils.condition(x)
        self.x = x
        self._center = center
        self._scale = scale
        self.y = y
        self.ybin, self.classNames = utils.makeBinary(y)
        
        self.regularization = self.regScale/(x**2).mean()
        M = np.dot(x.T, x)
        ind = np.arange(len(M))
        M[ind, ind] += self.regularization
        xTy = np.dot(x.T, self.ybin)
        self.w = np.dot(np.linalg.inv(M), xTy)
        
    def pred(self, x):
        """
        """
        # Condition the data, using the same conditioning as for the training
        # set
        x, dummy1, dummy2 = utils.condition(x, self._center, self._scale)
        classInd = np.dot(x, self.w).argmax(1)
        return utils.columnVector(self.classNames[classInd])
    
def regularizationPath(x, y, classifier, regScale0, stepSize=2.,
                       minRegScale=1e-11, maxIter=100):
    """
    Perform a very basic regularization path, with losses counted as 0/1
    loss
    """
    regScales = []
    losses = []
    stepSize = float(stepSize)
    regScale = regScale0
    iIter = 0
    done = False
    while (iIter < maxIter) and not done:
        
        classifier.fit(x, y, regScale=regScale)
        ypred = classifier.pred(x)
        loss = (ypred != y).mean()
        losses.append(loss)
        regScales.append(regScale)
        print iIter, regScale, loss
        iIter += 1
        regScale /= stepSize
        if regScale < minRegScale:
            done = True
            
    return np.array(losses), np.array(regScales)
    
if __name__ == '__main__':
    
    x, y = pickle.load(open('data_train.p','r'))
    xtest, ytest = pickle.load(open('data_test.p','r'))
    