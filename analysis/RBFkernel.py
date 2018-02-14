# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:43:24 2016

@author: ibackus
"""

import numpy as np

import utils



class RBFkernel():
    """
    RBFkernel(x, bandwidthScale=0.3, nCheckBandwidth=100)
    Makes a callable radial basis function kernel (feature map) defined
    by:
    
        .. math::
        
            K(x') = \\exp \\left( -\\frac{(x' - x)^2}{2 \\sigma^2} \\right)
            
    Where :math:`x', x` are the data points to be mapped and the input dataset.
    :math: `\\sigma` is the bandwidth, estimated by randomly selecting points
    in x and calculating the mean of the distance between them.
    
    This has been decently well optimized.
    
    Parameters
    ----------
    x : array
        Feature array to use this on
    bandwidthScale : float
        Amount to scale bandwidth by.
    nCheckBandwidth : int
        Number of data points to estimate distance between for bandwidth est.
    
    Methods
    -------
    __call__(x)
        Return performMap(x)
    performMap(x)
        Map x according to the kernel
    setupBandwidth(scale=None, nCheckBandwidth=None)
        Estimate a good bandwidth to use
    """
    def __init__(self, x, bandwidthScale=0.3, nCheckBandwidth=100, 
                 scalarOffset=False):
        
        self.x = x
        self.x2sum = (x**2).sum(1)
        n0, d = x.shape
        self.n0 = n0
        self.d = d
        self.setupBandwidth(bandwidthScale, nCheckBandwidth)
        self.scalarOffset = scalarOffset
    
    def __call__(self, x):
        """
        Map x using the RBFkernel feature map
        """
        return self.performMap(x)
        
    def setupBandwidth(self, scale=None, nCheckBandwidth=None):
        """
        Set up the bandwidth by randomly selecting nCheckBandwidth points
        and taking the average distance between them then scaling by scale
        """
        # Access data
        x = self.x
        n, d = x.shape
        
        # Parse Arguments
        if nCheckBandwidth is not None:
            
            self.nCheckBandwidth = nCheckBandwidth
            
        if scale is not None:
            
            self.bandwidthScale = scale
            
        if self.nCheckBandwidth > (n - 1):
            
            self.nCheckBandwidth = (n - 1)
            
        # Get average distance between n random points
        batch1 = utils.randBatch(n, self.nCheckBandwidth)[0]
        batch2 = utils.randBatch(n, self.nCheckBandwidth)[0]
        d = np.linalg.norm(x[batch1] - x[batch2], axis=-1)
        self.setBandwidth(d.mean() * scale)
        
    def setBandwidth(self, bandwidth):
        """
        Set the bandwidth
        """
        self.bandwidth = bandwidth
        self._lengthscale = 1.0/(2*self.bandwidth**2)
        
    def performMap(self, x):
        """
        Map x using the RBFkernel feature map
        """
        # Load data from self
        x0 = self.x
        x2sum0 = self.x2sum        
        # Calculate the square of distances between point x[i], x0[j]
        x2sum = (x**2).sum(1)        
        # Calculate distances squared
        # This is complicated, but is equiv to:
        # D2_transpose = -2 x0.dot.x_transpose
        # Add x2sum to the rows of D2_transpose (cols of D2)
        # D2 = D2.T
        # Add x2sum0 to rows of D2
        D2 = (-2*np.dot(x0, x.T) + x2sum).T        
        D2 += x2sum0 # Add x2sum0 to the rows
        # Exponentiate to perform mapping
        H = np.exp(-D2 * self._lengthscale)
        # Scalar offset
        if self.scalarOffset:
            
            H = utils.addColumns(H, value=1.)
            
        return H