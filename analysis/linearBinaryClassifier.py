#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Defines a simple class to classify data labeled by binary y

Created on Fri Dec  2 15:52:23 2016

@author: ibackus
"""
import numpy as np
from scipy.interpolate import interp1d

class LinearBinaryClassifier():
    """
    A simple class for deciding basic classification of binary data along one 
    dimension
    """
    def __init__(self, y, ypred, npts=400):
        """
        """
        self.npts = npts
        self.y = y
        mask = (y == 0)

        self.ymin = ypred.min()
        self.ymax = ypred.max()
        
        # Split up ypred into the 2 classes
        self.ypred = [ypred[mask], ypred[~mask]]
        # 
        self.ntot = len(ypred)
        self.ny = [len(yi) for yi in self.ypred]
        # Get spline functions for how many are less than y
        self.ysorted = [np.sort(yi) for yi in self.ypred]
        self.nless = [interp1d(self.ysorted[i], np.arange(self.ny[i]), \
            kind='nearest', fill_value='extrapolate', bounds_error=False, \
            assume_sorted=False) for i in range(2)]
        self.setupThreshold()
    
    def _scaleprob(self, p, pmin):
        """
        Re-scales the probabilities such that p.max() is preserved but 
        p <= pmin goes to zero
        
        to avoid some edge cases, pmin >= p.max() sets pmin=0
        """
        pmax = p.max()
        if pmin >= pmax:
            pmin = 0.
        p = pmax * (p - pmin)/(pmax - pmin)
        p[p < 0] = 0
        return p
        
    def setupThreshold(self):
        """
        setup the threshold and confidences
        """
        self.yline = np.linspace(self.ymin, self.ymax, self.npts)
        # Number of 1s less than y
        n1lt = self.nless[1](self.yline)
        # number of 0s greater than y
        n0gt = self.ny[0] - self.nless[0](self.yline)
        # Prediction error as a function of threshold
        error = n0gt + n1lt
        self.iThreshold = error.argmin()
        self.threshold = self.yline[self.iThreshold]
        # approx prob (really the CDF) of being in 1
        prob1 = n1lt/(n1lt + n0gt)
        prob = [1-prob1, prob1]
        # Rescale to get the 'confidence'
        i0 = self.iThreshold
        i1 = min(self.npts-1, i0+1)
        boundaryProbs = [0.5 * (probi[i0] + probi[i1]) for probi in prob]
        conf = [self._scaleprob(prob[i], boundaryProbs[i]) for i in range(2)]
        # Set up confidence splines
        self.conf = [interp1d(self.yline, confi, kind='nearest', \
            fill_value='extrapolate', bounds_error=False) for confi in conf]
        
        self._prob = prob
        self._confarrays = conf
        self._error = error
        
    def classify(self, ypred):
        """
        classify ypred.  ypred > self.threshold gives 1, else gives 0
        """
        ypred = np.asarray(ypred)
        yclass = np.zeros(ypred.shape)
        yclass[ypred > self.threshold] = 1
        return yclass
    
    def confidence(self, ypred):
        """
        Assign a confidence of the classification of ypred
        """
        ypred = np.asarray(ypred)
        yclass = self.classify(ypred)
        conf = np.zeros(ypred.shape)
        mask = (yclass == 0)
        conf[mask] = self.conf[0](ypred[mask])
        conf[~mask] = self.conf[1](ypred[~mask])
        return conf
    