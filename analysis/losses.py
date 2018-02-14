# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 13:05:35 2016

@author: ibackus
"""
import numpy as np
import utils

class lossClassifier():
    """
    lossClassifier(self, scheme='linear squared multiclass', **kwargs)
    Defines a class that constructs groups of loss functions, gradients of
    loss functions, and classifiers for different schemes.
    
    **kwargs are used by the different schemes.
    
    Schemes
    -------
    'linear squared multiclass'
        A linear model using one vs all classification and squared loss
    'softmax'
        Softmax for multiclass prediction
    'linear binary'
        Binary class.  All labels should be 0 or 1
        
    * kwargs :
    
        * **classNames** : None or list of names for all the classes
        
        * **regularization** : float, regularization to use
        
        * **lamScale** : float (overrides regularization) Calculate the 
            regularization as lamScale/x**2.mean() (only if x is supplied)
    """
    available_schemes = ('linear squared multiclass', 'softmax', 'linear binary')
    
    def __init__(self, scheme='linear squared multiclass', x=None, y=None, **kwargs):
        """
        """
        self.kwargs = kwargs
        
        if scheme not in self.available_schemes:
            
            raise ValueError, 'Unrecognized scheme.  Available schemes: {0}'\
            .format(self.available_schemes)
            
        self.scheme = scheme
        # Set up the classifier to use
        self.classNames = kwargs.get('classNames', None)
        # Setup regularization
        self._setupRegularization(x)
            
        if scheme == 'linear squared multiclass':
            
            self.classifier = self._linearOneVsAll
            # Set up the losses to use
            self._loss = self._linearSquaredLoss
            self._gradLoss = self._gradLinearSquaredLoss
            
        elif scheme == 'linear binary':
            
            self.classifier = self._linearBinaryPred
            self._loss = self._linBinSquaredLoss
            self._gradLoss = self._gradLinearSquaredLoss
            
        elif scheme == 'softmax':
            
            self.classifier = self._softmaxClassify
            self._loss = self._softmaxLogLoss
            self._gradLoss = self._softmaxGradLoss
            
    def loss(self, x, y, w, dataWeights=None):
        """
        Calculate loss, including (possible) regularization
        """
        # Calculate loss
        L = self._loss(x, y, w, dataWeights)
        # Regularize loss
        L += (w**2).sum() * self.regularization
        return L
        
    def gradLoss(self, x, y, w, dataWeights=None):
        """
        Calclate the gradient of the loss (with regularization if included)
        """
        gradL =  self._gradLoss(x, y, w, dataWeights)
        gradL += 2 * self.regularization * w
        return gradL
            
    def _setupRegularization(self, x):
        
        self.regularization = self.kwargs.get('regularization', 0.)
        if (x is not None) and ('lamScale' in self.kwargs):
            
            self.regularization = self.kwargs['lamScale'] / (x**2).mean()
            
    # Linear binary functions
    def _linearBinaryPred(self, x, w):
        return linearBinaryPred(x, w)
    def _linBinSquaredLoss(self, x, y, w, dataWeights=None):
        return linBinSquaredLoss(x, y, w)

    # Linear multiclass functions
    def _linearOneVsAll(self, x, w):
        return linearOneVsAll(x, w, self.classNames)
    def _linearSquaredLoss(self, x, y, w, dataWeights=None):
        return linearSquaredLoss(x, y, w)
    def _gradLinearSquaredLoss(self, x, y, w, dataWeights=None):
        return gradLinearSquaredLoss(x, y, w)
    # Softmax functions
    def _softmaxClassify(self, x, w):
        return softmaxClassify(x, w, self.classNames)
    def _softmaxGradLoss(self, x, y, w, dataWeights=None):
        return softmaxGradLoss(x, y, w, dataWeights)
    def _softmaxLogLoss(self, x, y, w, dataWeights=None):
        return softmaxLogLoss(x, y, w, dataWeights)


# -------------------------------------------------------------------------
# LINEAR BINARY
# -------------------------------------------------------------------------
def linearBinaryPred(x, w):
    """
    """
    ypred = np.dot(x, w)
    mask = ypred > 0.5
    ypred[mask] = 1
    ypred[~mask] = 0
    
    return utils.columnVector(ypred)

def linBinSquaredLoss(x, y, w):
    """
    """
    ypred = utils.columnVector(np.dot(x, w))
    L = ((ypred - y)**2).mean()
    return L
# -------------------------------------------------------------------------
# LINEAR MULTICLASS
# -------------------------------------------------------------------------
def linearOneVsAll(x, w, classNames=None):
    """
    For a multiclass (one vs all) linear model predict y (as x.dot.w) and
    take return the argmax() of that.  classNames gives the names of the
    different classes.  If None, the are 0, 1, 2, ...
    
    Parameters
    ----------
    x : array
        Feature vectors.  shape (n x d)
    w : array
        Weights, shape (d x nClass)
    """
    if classNames is None:
        
        classNames = np.arange(w.shape[-1])
        
    ybinpred = np.dot(x, w)
    ind = ybinpred.argmax(1)
    return classNames[ind]
    
    
def linearSquaredLoss(x, y, w):
    """
    For a linear model return the squared loss averaged across data points
    and summed across classes
    
    Parameters
    ----------
    x : array
        Feature vectors.  shape (n x d)
    y : array
        Binary labels.  shape (n x nClass).  y[i,j] = 1 for data[i] belonging
        to class[j], else it equals 0.
    w : array
        Weights, shape (d x nClass)
    """
    # Sum along the features and take average across data points
    ypred = np.dot(x, w)
    L = 0.
    L = ((ypred - y)**2).sum(1).mean()
    return L
    
def gradLinearSquaredLoss(x, y, w):
    """
    Returns the gradient (with respect to weights) of the a linear model
    squared loss
    
    Parameters
    ----------
    x : array
        Feature vectors.  shape (n x d)
    y : array
        Binary labels.  shape (n x nClass).  y[i,j] = 1 for data[i] belonging
        to class[j], else it equals 0.
    w : array
        Weights, shape (d x nClass)
    """
    N = len(x)
    gradL = (2.0/N) * np.dot(x.T, np.dot(x, w) - y)
    return gradL

# -------------------------------------------------------------------------
# SOFTMAX
# -------------------------------------------------------------------------
def softmaxClassify(x, w, classNames=None):
    """
    Classify x, given weights w, according to the softmax scheme.
    x is shape (n x d)
    w is shape (d x nClass-1)
    (optional) classNames is length nClass
    """
    if classNames is None:
        nClass = w.shape[1] + 1
        classNames = np.arange(nClass)
    
    P = softmaxProb(x, w)
    return classNames[P.argmax(1)]
                      
def softmaxGradLoss(x, y, w, dataWeights=None):
    """
    """
    n = len(x)
    nClass = w.shape[1] + 1
    P = softmaxProb(x, w)
    # Weights can be applied here since the sums over data points are handled
    # via terms with x.T dotted into another matrix
    if dataWeights is not None:
        x = x*dataWeights
    xTp = np.dot(x.T, P)
    xTc = _xTc(x, y)
    return (xTp[:,0:nClass-1] - xTc)/n
    
def softmaxProb(x, w):
    """
    P[i, j] is the probability of belonging to class j given x[i], w[:, j]
    """
    N, m = w.shape
    k = m + 1
    E = np.exp(np.dot(x, w))
    A = 1./(1 + E.sum(1))
    A = utils.columnVector(A)    
    P = np.dot(A, np.ones([1, k]))    
    P[:, 0:-1] *= E
    
    return P
    
def _xTc(x, y):
    """
    A term used in updating the weights for the softmax gradient descents
    
    Parameters
    ----------
    x : array
        Feature vectors.  shape (n x d)
    y : array
        Binary labels.  shape (n x nClass).  y[i,j] = 1 for data[i] belonging
        to class[j], else it equals 0.
    """
    A = utils.columnVector(1 - y[:, -1])
    B = y[:, 0:-1]
    C = A*B
    return np.dot(x.T, C)
    
def softmaxLogLoss(x, y, w, dataWeights=None):
    """
    Given compare y to the prediction from x, w.  Calculates log loss and
    0/1 loss
    """
    P = softmaxProb(x, w)
    logP = np.log(P[y.astype(bool)])
    if dataWeights is not None:
        logP *= dataWeights.flatten()
    logloss = -logP.mean()
#    logloss = -np.log(P[y.astype(bool)]).mean()
#    if dataWeights is not None:
#        logloss = -(np.log(P[y.astype(bool)]) * dataWeights.flatten()).sum()
#    else:
#        logloss = -np.log(P[y.astype(bool)]).mean()
    return logloss
