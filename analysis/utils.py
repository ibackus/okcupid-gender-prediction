# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 17:48:31 2016

@author: ibackus
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def makeBinary(y, classNames = None):
    """
    Converts a vector of labels into a binary matrix
    
    ybin[i,j] = 1 for y[i] in class j
    """
    if classNames is None:
        
        classNames = np.unique(y)
        
    nClass = len(classNames)
    ybin = np.zeros([len(y), nClass])
    y = y.flatten()
    for i, className in enumerate(classNames):
        mask = (y == className)
        ybin[mask, i] = 1

    return ybin, classNames

def condition(x, center=None, scale=None):
    """
    Conditions x by subtracing off the mean and normalizing by the std.dev 
    (along the first axis).
    
    if the standard deviation is zero, scale = 1
    
    Parameters
    ----------
    x : array
        Array to conditions.  shape (n, d)
    
    center: array or float
        Quantity to shift by (defaults ot the mean).  either array length d
        or a float
    scale : array or float
        Qantity to scale (After shifting) by.  array length d or a float
        
    Returns
    -------
    x_conditioned : array
        A conditioned version of x
    center, scale : array/floats
        center and scale used
    """
    if center is None:
        
        center = x.mean(0)
        
    if scale is None:
        
        scale = np.asarray(x.std(0))
        scale[scale == 0] = 1.
        
    x = (x - center)/scale
    return x, center, scale

def randBatch(N, batchsize):
    """
    Create a list arrays of randomly selected indices to select batches of 
    an array length N
    """
    ind = np.random.rand(N).argsort()
    if batchsize == 1:
        return ind[:,None]
    nBatch = int(np.ceil(float(N)/batchsize))
    batches = np.array_split(ind, nBatch)
    return batches
    
def arrayBatch(x, batchsize=None):
    """
    Creates an iterator to return sub-batches (slices) for an array.
    batchsize defaults to len(x)
    
    Examples
    --------
    >>> x = np.random.rand(100)
    >>> for a in arrayBatch(x, 12):
    >>>     print x[a]
    """
    if batchsize is None:
        
        batchsize = len(x)
        
    for i in xrange(0, len(x), batchsize):
        #yield x[i:i + batchsize]
        yield slice(i, i+batchsize)

def gridPlot(ims, imshape=[28, 28], plotOne=False, grid=None):
    """
    Plots a grid of N images.  ims is a 2D or 3D array. len(ims) = number of
    ims to plot
    """
    if plotOne:
        
        if len(ims) != 1:
            
            ims = ims[None, :]
        
    nImage = len(ims)
    
    if grid is None:
        
        N = int(np.ceil(np.sqrt(nImage)))
        N2 = int(np.ceil(nImage/float(N)))
    
        plt.clf()
        fig = plt.gcf()    
        grid = ImageGrid(fig, 111, nrows_ncols=(N, N2), axes_pad=0.05)
    
    for i in range(nImage):
        
        im = ims[i].reshape(imshape)
        grid[i].imshow(im, interpolation='none', cmap='gray')
        grid[i].text(2, 2, i, color='red', ha='center', va='center')
        
    plt.draw()
    plt.show()
    return grid

def savefig(savename):
    """
    A wrapper for matplotlib.pyplot.savefig which echos what filename the
    figure was saved to
    """
    plt.savefig(savename)
    print 'Figure saved to', savename

def fracDiff(x1, x2):
    """
    Calculates the fractional difference between x1 and x2
    equal to 2(x1 - x2)/(x1 + x2)
    """
    
    xmean = 0.5 * (x1 + x2)
    xdiff = x1 - x2
    return xdiff/xmean

def addColumns(x, value=1., N=1, end=True):
    """
    Adds N columns to x at the end (if end) or at the beginning and inserts 
    value into them.
    
    Does NOT copy x
    """
    nRow, nCol = x.shape
    x2 = np.zeros([nRow, nCol + N])
    if end:
        x2[:, :-N] = x
        x2[:,-N:] = value
    else:
        x2[:,N:] = x
        x2[:,0:N] = value
    return x2

def splitdata(x, y, nTrain, nTest, seed=None):
    """
    Randomly splits the (data, labels) (x,y) into training, test, and validation
    sets.  nTrain and nTest are put into training, test, and the rest are 
    put in the validation set.  seed can be used to seed the random number
    generator
    
    Returns
    -------
    data : dict
        A dict with three keys 'test', 'train', and 'validaton'
        The values are stored as tuples of (x,y)
    """
    N, d = x.shape
    assert(len(y) == N)
    # Generate randomly shuffled indices
    if seed is not None:
        np.random.seed(seed)
    ind = np.random.rand(N).argsort()
    trainInd = ind[0:nTrain]
    testInd = ind[nTrain:nTrain + nTest]
    validationInd = ind[nTrain + nTest:]
    train = (x[trainInd], y[trainInd])
    test = (x[testInd], y[testInd])
    validation = (x[validationInd], y[validationInd])
    
    return {'test': test, 'train': train, 'validation': validation}
    

def lambda0(x, y, norm=None):
    """
    Returns an estimate of smallest value of regularization lambda for which 
    the solution w is entirely zero (for a linear, regularized model)
    
    Parameters
    ----------
    x : array
        data
    y : array
        labels
    norm : (optional)
        which norm to use.  See numpy.linalg.norm ord
    """
    return 2 * np.linalg.norm(np.dot(x.T, y-y.mean()), ord=norm)
    
def linearResiduals(x, y, w0, w):
    """
    Calculate the square of the residuals of a linear model
    
    result = (y - ypred)**2
    """
    n, d = x.shape
    y = columnVector(y)
    w = columnVector(w)
    ypred = columnVector(linearPredict(x, w, w0))
    return (y - ypred)**2
    
    
def linearPredict(x, w, w0):
    """
    Given a set of weights, predict y from x for a linear model.
    
    Parameters
    ----------
    x : array
        data points, shape = n x d
        n is number of points
        d is the dimensionality (number of features)
    w : array
        weights, shape = d x 1
    w0 : float
        Scalar offset
    
    Returns
    -------
    y : array
        Predicted y, shape = n x 1
    """
    return (w0 + np.dot(x, columnVector(w)))
    
def columnVector(x):
    """
    A convenience utility to convert an array that is 0D, 1D, or 2D into a 
    column vector.  Row vectors will get transposed.  Scalars become 1x1 
    arrays
    """
    if x is None:
        
        return x
        
    x = np.asarray(x)
    ndim = np.ndim(x)
    if ndim == 0:
        return x.reshape([1,1])
    elif ndim == 1:
        return x[:,None]
    elif ndim == 2:
        
        nrows, ncols = x.shape
        if (nrows > 1) and (ncols > 1):
            
            raise ValueError, "cannot turn this into a column vector"
            
        if nrows == 1:
            
            return x.T
            
        else:
            
            return x
            
    else:
        
        raise ValueError, "x is not a vector"