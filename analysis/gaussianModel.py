#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Contains functions and class definitinos for supervised training of a 
multiclass gaussian model

Created on Wed Nov 30 20:56:56 2016

@author: ibackus
"""
import numpy as np
import utils

class GaussianModel():
    """
    GaussianModel() fits multiple gaussians to the given data using a VERY
    simple method.
    
    k multivariate normal distributions are fit by looping over the data
    in the k classes and fitting a multivariate normal.  predictions are
    done by returning what the maximum probability
    
    Confidence is defined as (maximum probability)/(sum of probabilities)
    for a given data point, which works under the assumption that every 
    data point belongs to one of the classes.
    """
    def __init__(self):
        
        pass
    
    def fit(self, x, y, regScale=0.0):
        """
        """
        self.regScale = regScale
        self.classNames = np.unique(y)
        self.nClass = len(self.classNames)
        # Split the xs for all the different classes
        xs = [x[y.flatten() == className] for className in self.classNames]
        # Generate pdfs for all the gaussians
        self.pdfs = [makePDF(xi, regScale=0.0) for xi in xs]
    
    def pred(self, x):
        """
        Predict the class of x
        """
        probs = self.prob(x)
        classInd = probs.argmax(1)
        return utils.columnVector(self.classNames[classInd])
        
    def prob(self, x):
        """
        Probability of x belonging to the different classes
        for P = prob(x),
        P[i,j] is prob of x[i] belonging to classj
        """
        probs = np.array([pdf(x) for pdf in self.pdfs])
        return probs.T
    
    def confidence(self, x):
        """
        Confidence is defined as (maximum probability)/(sum of probabilities)
        for a given data point, which works under the assumption that every 
        data point belongs to one of the classes.
        """
        P = self.prob(x)
        psum = P.sum(1)
        pnorm = P.max(1)/P.sum(1)
        pnorm[psum == 0] = 0
        return utils.columnVector(pnorm)
    
def multivariateNormalPDF(covar, mean, regScale=0.0):
    """
    Generates a function f(x) which is a multivariate normal distribution.
    x is assumed to store the vectors along the rows (ie x is a row vector or
    a matrix composed of row vectors)
    """
    mean = utils.columnVector(mean).T
    # Regularize
    if regScale != 0:
        ind = np.arange(len(covar))
        regularization = abs(covar[ind, ind]).mean() * regScale
        covar[ind, ind] += regularization
    # Calculate terms
    covarInv = np.linalg.inv(covar)
    det = np.linalg.det(2 * np.pi* covar)
    norm = 1./np.sqrt(det)
    
    def PDF(x):
        """
        x is assumed to store the vectors along the rows (ie x is a row vector or
        a matrix composed of row vectors)
        """
        if len(x) > 1:
            
            return np.array([PDF(x1[None, :]) for x1 in x])
            
        return float(norm \
            * np.exp(-0.5 * np.dot((x-mean), np.dot(covarInv, (x-mean).T))))
    
    return PDF
    
def makePDF(x0, regScale=0.0):
    """
    """    
    center = x0.mean(0)
    x0centered = x0 - center
    covar = np.dot(x0centered.T, x0centered)/(len(x0) - 1.)
    PDF = multivariateNormalPDF(covar, center, regScale)
    return PDF