#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:06:52 2016

@author: ibackus
"""

import numpy as np
from scipy import stats

import utils


class kmeans():
    """
    """
    def __init__(self, x, nCluster=2, maxIter=100, seed=None, xtol=1e-8,
                 verbose=True):
        
        # Store data
        self.x = x
        self.x2sum = (x**2).sum(1)[:,None]
        self.dNorm = x.std(0)
        self.dNorm[self.dNorm == 0] = 1.0
        self.nCluster = nCluster
        self.maxIter = maxIter
        self.seed = seed
        self.xtol = xtol
        self.verbose = verbose
        
    def _assignCentroid(self):
        """
        Assigns the points in self.x to centroids in self.centroid and
        calculates the error
        """
        # Assign centroids
        d2 = dist2(self.x, self.centroids, self.x2sum)
        self.clusterInd = d2.argmin(1)
        # Calculate the error
        d2min = d2[np.arange(len(d2)), self.clusterInd]
        error = d2min.mean()
        self.error.append(error)
        
    def initFit(self, centroids=None, dataWeights=None):
        """
        Initialize a fit
        """
        if centroids is None:
            # Initialize centroid guesses
            np.random.seed(self.seed)
            ind = np.random.choice(len(self.x), self.nCluster)
            self.centroids = self.x[ind]
            
        else:
            self.centroids = centroids
            
        if dataWeights is None:
            
            dataWeights = np.ones([len(self.x), 1])
            
        else:
            
            dataWeights = utils.columnVector(dataWeights)
            
        self.dataWeights = dataWeights
        
        self.error = []
        self.oldcentroids = self.centroids.copy()
        self.centroidMaxChange = []
        self._assignCentroid()
    
    def iterate(self):
        """
        Perform one iteration of the fit
        """
        for i in range(self.nCluster):
            
            mask = (self.clusterInd == i)
            
            if np.any(mask):
                
                # centroids are a weighted average of x positions
                w = self.dataWeights[mask]
                w /= w.sum()
                self.centroids[i] = (w * self.x[mask]).sum(0)
            
        self._assignCentroid()
        # Calculate the amount the centroids have moved
        deltaCent = (self.centroids - self.oldcentroids)
        for i in range(len(deltaCent)):
            
            deltaCent[i] /= self.dNorm
            
        self.centroidMaxChange.append(abs(deltaCent).max())
        self.oldcentroids = self.centroids.copy()
        
        if self.verbose:
                print '(max centroid change, error): ', \
                self.centroidMaxChange[-1], self.error[-1]
                
        if np.isnan(self.centroidMaxChange[-1]):
            
            raise RuntimeError, 'Fit failed!'
        
    def fit(self, centroids=None, dataWeights=None):
        """
        Run the fit
        """
        self.initFit(centroids, dataWeights)
        self.iIter = 0
        self.converged = False
        
        
        # Do one iteration
        self.iterate()
        self.iIter += 1
        
        # Now do the remaining ones
        while (self.iIter < self.maxIter) and (not self.converged):
            
            self.iterate()
            
            if self.centroidMaxChange[-1] < self.xtol:
                
                self.converged = True
                
            self.iIter += 1
            
        print 'Final error: ', self.error[-1]
        
        
        # remove clusters with nobody in them
        clusterInd = self.clusterInd
        present = np.unique(clusterInd)
        keep = np.zeros(self.nCluster, dtype=bool)
        keep[present] = 1
        if not np.all(keep):
            print "dropping unused clusters"
            self.centroids = self.centroids[keep]
            self.nCluster = keep.sum()
            self.clusterInd = self.predCluster(self.x)
            
    def predCluster(self, x):
        """
        Get the index of the centroids closest to the points in x
        """
        d2 = dist2(x, self.centroids)
        clusterInd = d2.argmin(1)
        
        return clusterInd
        
        
    def _pruneClusters(self):
        """
        remove clusters with nobody in them
        """
        clusterInd = self.clusterInd
        #clusterInd = self.predCluster(self.x)
        present = np.unique(clusterInd)
        keep = np.ones(self.nCluster, dtype=bool)
        for iCluster in range(self.nCluster):
            if iCluster not in present:
                keep[iCluster] = False
        
        self.centroids = self.centroids[keep]
        self.nCluster = keep.sum()
        self.clusterInd = self.predCluster(self.x)
        
        
    
    def labelClusters(self, y):
        """
        Label the clusters according to y
        
        Labels are determined by finding the most common label belonging to
        a given cluster
        
        Also stores what fraction of cluster members have that name
        """
        clusterNames = np.zeros(self.nCluster)
        clusterConfidence = np.zeros(self.nCluster)
        
        for i in range(self.nCluster):
            
            mask = self.clusterInd == i
            y0 = y[mask]
            ny = len(y0)
            clusterNames[i], count = stats.mode(y[mask])
            
            if count != 0:
                clusterConfidence[i] = float(count)/ny
            
        self.clusterNames = clusterNames
        self.clusterConfidence = clusterConfidence
    
    def pred(self, x):
        """
        Predict the class x belongs to (only works if labelClusters has 
        been performed)
        """
        clusterInd = self.predCluster(x)
        ypred = self.clusterNames[clusterInd]
        return utils.columnVector(ypred)
        
    def confidence(self, x):
        """
        Confidence is given by the cluster confidence (what fraction of 
        elements in the cluster have the cluster name)
        """
        clusterInd = self.predCluster(x)
        conf = self.clusterConfidence[clusterInd]
        return utils.columnVector(conf)
    
def dist2(x, y, x2sum=None):
        """
        Calculate the square of the distances between the vectors in y and in
        x.  Vectors are stored along the rows.
        
        x2sum is (optionally) x**2.sum(1)
        
        returns D2
        
        D2[i,j] = ((x[i] - y[j])**2).sum()
        """
        if len(x) == x.size:
            
            x = utils.columnVector(x)
            
        if len(y) == y.size:
            
            y = utils.columnVector(y)
            
        if x2sum is None:
            
            x2sum = (x**2).sum(1)[:,None]
            
        else:
            
            x2sum = utils.columnVector(x2sum)
        
        y2sum = (y**2).sum(1)[:, None]
        D2 = -2 * np.dot(x, y.T)
        D2 = (D2.T + y2sum).T
        D2 += x2sum
        return D2

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    # Generate mock data
    nClass = 6
    nPoints = 1000
    
    centroids = np.zeros([nClass, 2])
    r = 10.
    theta = np.linspace(0, 2*np.pi, nClass+1)[0:-1]
    centroids[:, 0] = r*np.cos(theta)
    centroids[:, 1] = r*np.sin(theta)
    
    x = np.random.randn(nPoints, 2)
    classInd = np.random.choice(nClass, nPoints)
    for i in range(nClass):
        x[classInd == i] += centroids[i]
    
    def plotclusters(x, clusterInd, centroids=None):
        
        nCluster = clusterInd.max() + 1
        
        for clusterNum in range(nCluster):
            mask = clusterInd == clusterNum
            plt.plot(x[mask][:,0], x[mask][:, 1], 'o')
            
        if centroids is not None:
            plt.plot(centroids[:, 0], centroids[:, 1], 'o', markersize=10)
            
    km = kmeans(x, nClass)
    km.fit()
    plt.clf()
    plotclusters(km.x, km.clusterInd, km.centroids)
    
    plt.plot(centroids[:, 0], centroids[:, 1], 'v', markersize=10)