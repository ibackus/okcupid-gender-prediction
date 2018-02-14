# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:57:49 2016

@author: ibackus
"""
import numpy as np
import sys

import utils
import losses

class lossTracker():
    """
    Track losses during a fit
    """
    def __init__(self, parent, w, verbose=True):
        
        self.parent = parent
        self.verbose = verbose
        self.fitloss = {}
        
        for dataname in parent.datasets.keys():
            
            self.fitloss[dataname] = {'loss': [], 'binaryLoss': []}
        
        self.updateLosses(w)
        
    def updateLosses(self, w):
        """
        Calculate losses given the current weights and store in the fitloss
        dict
        """
        parent = self.parent
        dataWeights = None
        if hasattr(parent, 'dataWeights'):
            dataWeights = parent.dataWeights
        for name, dataset in parent.datasets.iteritems():
            
            x, y, ybin = dataset
            L, binaryLoss = parent.loss(x, y, ybin, w=w, dataWeights=dataWeights)
            
            if self.verbose:
                print '{0} loss: {1}'.format(name, L)
                print '{0} 0/1 loss: {1}'.format(name, binaryLoss)
                sys.stdout.flush()
            self.fitloss[name]['loss'].append(L)
            self.fitloss[name]['binaryLoss'].append(binaryLoss)
        
        if self.verbose:
            print ''

class fitMulticlassBase(object):
    """
    Weights (self.w) are just extra arguments to be passed to loss functions.
    For linear models, they should be implemented as normal weights
    """
    def __init__(self, learningRate=None, lossScheme='linear squared multiclass', \
        miniBatchSize=10, featureMapFcn=None, classNames=None, lamScale=0.0,
        verbose=True, arrayBatchSize=None, nLossCheck=None):
        
        if arrayBatchSize is None:
            print "Warning.  arrayBatchSize is None.  For large data sets this can fail"
        self.nLossCheck = nLossCheck
        self.arrayBatchSize = arrayBatchSize
        self.lossScheme = lossScheme
        self.lamScale = lamScale
        self.miniBatchSize = miniBatchSize
        self._learningRate = learningRate
        self.classNames = classNames
        self.datasets = {}
        self.verbose = verbose
        self.setFeatureMap(featureMapFcn)
        self.setLearningRate(learningRate)
        
    def results(self, extrakeys=[]):
        """
        Return a dict of results.  Great for pickling
        """
        out = {}
        keys = ['fitloss', 'lossScheme', 'nLossCheck', 'nClass', \
            'miniBatchSize', 'd', 'classNames', 'w', 'regularization']
        keys.extend(extrakeys)
        
        for key in keys:
            
            out[key] = self.__dict__.get(key, None)
            
        return out
            
    def setFeatureMap(self, featureMapFcn):
        """
        Set the feature map to use ( a function that maps x to different
        features).  If None, no feature map is used
        """
        # Set feature map
        if featureMapFcn is None:
            self.featureMap = lambda x: x
        else:
            self.featureMap = featureMapFcn
            
    def setLearningRate(self, learningRate):
        """
        Set the learning rate.  learningRate can be a callable function (as
        a function iIter) or a constant.  This sets the behavior of self.eta()
        """
        # Set learning rate
        if not callable(learningRate):
            self.eta = lambda x : learningRate
        else:
            self.eta = learningRate
            
    def _initdata(self, x, y, lamScale=None, miniBatchSize=None, dataWeights=None):
        """
        Ininitialize data for a fit
        """
        self.addDataset(x, y, 'training')
        if lamScale is not None:
            self.lamScale = lamScale
        
        if dataWeights is not None:
            # Normalize dataweights and scale by the total nuber of them
            dataWeights = len(dataWeights) * dataWeights/dataWeights.sum()
            
        self.dataWeights = dataWeights
        self._initLossClassifier()
        
        if miniBatchSize is not None:
            
            self.miniBatchSize = miniBatchSize
            
    
    def _initLossClassifier(self):
        """
        Initialize the loss functions and classifier to use
        
        See losses.lossClassifier()
        """
        # Look at a small subset of x, y and apply featureMap
        x = self.x
        y = self.y
        if self.arrayBatchSize is not None:
            
            ind = utils.randBatch(len(x), self.arrayBatchSize)[0]
            x = x[ind]
            y = y[ind]
            
        # Apply the feature map
        h = self.featureMap(x)
        # Load a classifier preset
        lossClassifier = losses.lossClassifier(scheme=self.lossScheme, \
            x=h, y=self.y, lamScale=self.lamScale, \
            classNames=self.classNames)
        self._lossFcn = lossClassifier.loss
        self.gradL = lossClassifier.gradLoss
        self.classifier = lossClassifier.classifier
        self.lossClassifier = lossClassifier
        if hasattr(self.lossClassifier, 'regularization'):
            
            self.regularization = self.lossClassifier.regularization
        
    def addDataset(self, x, y, name):
        """
        Add a data set with a name.  For instance, add a test set:
            self.addDataset(xtest, ytest, 'test')
        
        If self.classNames is None, set up the classNames from the labels
        in y
        """
        if name not in self.possibleDataSets:
            
            raise RuntimeError, "Unrecognized name.  Use of of {0}"\
                .format(self.possibleDataSets)
        
        # Set up classes
        if self.classNames is None:
            
            self.classNames = np.unique(y)
            
        self.nClass = len(self.classNames)
        
        if 'binary' in self.lossScheme:
            
            if self.nClass != 2:
                
                raise RuntimeError, "Cannot have binary classification for"\
                " {0} classes".format(self.nClass)
                
            ybin = y
            
        else:
            # Create binary y.  ybin[i,j] = 1 for y[i] == classNames[j], 0 else
            ybin = np.zeros([len(y), self.nClass])
            
            for i, className in enumerate(self.classNames):
                
                ybin[:, i] = (y == className).flatten()
            
        # Store data
        self.datasets[name] = (x, y, ybin)
        if name == 'training':
            
            self.x = x
            self.y = ybin
        
        
    def _initLosses(self):
        """
        Initializes the losses dict (fitloss) used to keep track of losses
        during the fit operation
        """
        self.lossTracker = lossTracker(self.datasets)
        
    def _updateLosses(self):
        """
        Calculate losses given the current weights and store in the fitloss
        dict
        """
        self.lossTracker.updateLosses(self.w)
        
    def loss(self, x, y, ybin, w=None, arrayBatchSize=None, nCheck=None,
             dataWeights=None):
        """
        Calculates the loss at points x, y, ybin
        """
        if dataWeights is None:
            
            dataWeights = np.ones([len(x), 1])/len(x)
            
        if arrayBatchSize is None:
            
            arrayBatchSize = self.arrayBatchSize
            
        L = 0.
        binaryLoss = 0.
        if w is None:
            
            w = self.w
        
        if nCheck is None:
            
            nCheck = self.nLossCheck
            
        if (nCheck is not None) and (nCheck < len(x)):
            
            # Select a random subset for checking the loss
            np.random.seed(0)
            batch = utils.randBatch(len(x), nCheck)[0]
            x = x[batch]
            y = y[batch]
            ybin = ybin[batch]
            dataWeights = dataWeights[batch]
            
            
        for slicer in utils.arrayBatch(x, arrayBatchSize):
            
            h = self.featureMap(x[slicer])
            y1 = y[slicer]
            ybin1 = ybin[slicer]
            dw = dataWeights[slicer]
            # Assume losses are MEANs
            L += len(h) * self._lossFcn(h, ybin1, w, dataWeights=dw)
            # Do the prediction with the feature map already applied
            ypred = self.classifier(h, w).flatten()
            binaryLoss += (ypred != y1.flatten()).sum()
            
        # Normalize by number of data points
        L /= len(x)
        binaryLoss /= len(x)
        return L, binaryLoss
        
    def pred(self, x, batch=True):
        """
        Predict the class of x according to the fit weights w using the 
        assigned classifier.  Calculate in batches (set by self.miniBatchSize)
        """
        if batch and len(x) > self.arrayBatchSize:
            # Calculate y_predicted in batches
            ypred = np.zeros(len(x))
            for slicer in utils.arrayBatch(x, self.arrayBatchSize):
                
                ypred[slicer] = self.pred(x[slicer]).flatten()
        
        else:
            
            if self.classifier is None:
                
                raise RuntimeError, "No classifier set.  Cannot predict"
                
            h = self.featureMap(x)
            ypred = self.classifier(h, self.w, self.classNames)
        
        return utils.columnVector(ypred)


def powerlawLearningRate(x, learningRateScale=1., kappa=0.6, tau0=1000):
    """
    Returns a learning rate function which decreases as a powerlaw in the
    number of iterations according to:
    
        .. math::
        
            \\eta = \\frac{\\eta_0}{(1 + k/\\tau_0)^{\\kappa}}
            
    Returns
    -------
    eta : function
        eta(iterationNum) returns the learning rate at that iteration
    """
    eta0 = learningRateScale/((x**2).mean())
    tau0 = float(tau0)
    
    def learningRate(iTot):
        eta = eta0/(1 + iTot/tau0)**kappa  
        return eta
    
    return learningRate
