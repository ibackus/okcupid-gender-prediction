#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:53:14 2016

@author: ibackus
"""

import numpy as np
import utils

class RandomMap():
    """
    RandomMap(self, d, k, seed=None) creates a random feature map (same as
    using a 1 layer deep neural network)
    
    Parameters
    ----------
    d : int
        Number of features in the data
    k : int
        Number of features to generate
    seed : number
        (optional) seed for numpy.random.seed when generating features
        
    Methods
    -------
    __call__(x)
        Return the mapping of x
    generateMap()
        Generate the random feature map
    """
    def __init__(self, d, k, seed=None, scalarOffset=False):
        
        self.d = d
        self.k = k
        self.seed = seed
        self.scalarOffset = scalarOffset
        self.generateMap()
        
    def __call__(self, x):
        
        H = np.dot(x, self.v)
        H[H<0] = 0.
        
        if self.scalarOffset:
            
            H = utils.addColumns(H, value=1.)
            
        return H
        
    def generateMap(self):
        
        np.random.seed(self.seed)
        self.v = np.random.rand(self.d, self.k)