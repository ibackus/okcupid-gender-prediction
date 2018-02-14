# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 19:38:35 2016

@author: ibackus
"""
import numpy as np
import sys
import cPickle as pickle

import utils
import fitutils
import convergenceTest

tempfilename = 'fitresults.p'

class SGD(fitutils.fitMulticlassBase):
    """
    """
    possibleDataSets = ('training', 'test', 'validation')
    
    def __init__(self, learningRate, lossScheme='linear squared multiclass', \
        miniBatchSize=10, featureMapFcn=None, classNames=None, lamScale=0.0,
        arrayBatchSize=None, nLossCheck=None):
        """
        Parameters
        ----------
        lossScheme : str
            Loss scheme to use.  See losses.lossClassifier() for options
            
        lossFcn : function
            Loss function to minimize.  Should be a function of x, y, w
            
        """
        super(SGD, self).__init__(learningRate, lossScheme, miniBatchSize, \
        featureMapFcn, classNames, lamScale, arrayBatchSize=arrayBatchSize, \
        nLossCheck=nLossCheck)
        
    def _initWeights(self, w=None):
        """
        Initialize the weights to self.w and the dimension of the weights
        to self.d
        
        Parameters
        ----------
        w : array-like
            (optional) Initial guess of the weights
        d : int
            (optional) Dimension of the weights (number of features)
        """
        if self.lossScheme == 'softmax':
            nCol = self.nClass - 1
        elif 'binary' in self.lossScheme:
            nCol = 1
        else:
            nCol = self.nClass
        if w is None:
                
            x1 = self.x[None, 0]
            h1 = self.featureMap(x1)
            d = h1.shape[1]
            self.w = self.w = np.zeros([d, nCol])
            self.d = d
            
        else:
            
            self.w = w.copy()
            self.d = len(w)
        
            
    def _initLosses(self):
        """
        Initializes the losses dict (fitloss) used to keep track of losses
        during the fit operation
        """
        self.lossTracker = fitutils.lossTracker(self, self.w)
        self.fitloss = self.lossTracker.fitloss
        
    def _updateLosses(self):
        """
        Calculate losses given the current weights and store in the fitloss
        dict
        """
        self.lossTracker.updateLosses(self.w)
        
    def results(self, extrakeys=[]):
        """
        Returns the results as a nice dict
        """
        out = super(SGD, self).results(extrakeys)
        return out
        
        
    def fit(self, x, y, miniBatchSize=None, w=None, maxIter=10, ftol=1e-3,
            lookback=3, minIter=2, lamScale=None, dataWeights=None):
        """
        Perform the stochastic gradient descent to minimize the loss function
        """
#        
        self._initdata(x, y, lamScale=lamScale, miniBatchSize=miniBatchSize,
                       dataWeights=dataWeights)
        # Initialize the while loop
        self._initWeights(w)
        self._initLosses()
        
        nPts = len(self.x)
        iData = 0
        conv = convergenceTest.ConvergenceTest(method='ftol', ftol=ftol, \
            lookback=lookback, minSteps=minIter)
        conv.addStep(self.fitloss['training']['loss'][-1])
        
        while (not conv.converged) and (conv.nSteps <= maxIter):
            
            print 'Step: ', conv.nSteps
            # Loop over randomly selected batches of the data
            batches = utils.randBatch(nPts, self.miniBatchSize)
            for iBatch, batch in enumerate(batches):
                
                x = self.x[batch]
                y = self.y[batch]
                h = self.featureMap(x)
                if dataWeights is not None:
                    dw = dataWeights[batch]
                else:
                    dw = None
                self.w -= self.eta(iData) * self.gradL(h, y, self.w, dataWeights=dw)
                iData += len(batch)
                
            self._updateLosses()
            conv.checkConvergence(self.fitloss['training']['loss'][-1])
            sys.stdout.flush()
            # Save results
            pickle.dump(self.results(), open(tempfilename, 'w'), 2)
        
        return