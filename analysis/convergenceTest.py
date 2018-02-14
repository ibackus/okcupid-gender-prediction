# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:34:35 2016

@author: ibackus
"""

class ConvergenceTest():
    """
    ConvergenceTest(method='ftol', xtol=None, ftol=1e-4, lookback=1, \
                    minSteps=2)
     
     A simple class for handling convergence tests.
     
     Parameters
     ----------
     method : str
         convergence test method to use.  Currently implemented are 'ftol' 
         (follow a scalar quantity like the loss)
     xtol : float
         (not implemented)
     ftol : float
         tolerance for fractional change in function (loss) tolerance
     lookback : int
         Compare current value (loss, argument vals, etc) to max of the previous
         lookback number of values
     minSteps : int
         Minimum number of steps before checking convergence
         
     Examples
     --------
     >>> conv = convergenceTest(ftol=1e-3, lookback=3)
     >>> maxIter = 20
     >>> while not conv.converged and conv.nSteps < maxIter:
     >>>    # Calculate losses
     >>>    # ...
     >>>    conv.checkConvergence(loss)
    """
    
    def __init__(self, method='ftol', xtol=None, ftol=1e-4, lookback=1,
                 minSteps=2):
        """
        """
        if method not in ('ftol'):
            
            raise ValueError, 'Unrecognized convergence test method {0}'\
            .format(self.method)
            
        self.method = method
        self.xtol = xtol
        self.ftol = ftol
        self.lookback = lookback
        self.minSteps = minSteps
        self.reset()
        
        
    def addStep(self, x):
        """
        Append a loss (without checking for convergence) or a function argument
        
        If method = 'ftol', append x as a loss
        """
        if self.method == 'ftol':
            
            self.loss.append(x)
            
        self.nSteps += 1
        
    def checkConvergence(self, loss):
        """
        Append a loss and check for convergence
        """
        self.addStep(loss)
        
        if (self.nSteps <= self.minSteps) or (self.nSteps <= self.lookback):
            
            return
        
        if self.method == 'ftol':
            
            self._ftolCheck()
            
        if self.converged:
            
            print 'Converged'
            
    def reset(self):
        """
        Reset to step 0...delete losses etc
        """
        self.loss = []
        self.funcargs = []
        self.nSteps = 0            
        self.converged = False
    
    def _ftolCheck(self):
        """
        Check if fractional change in losses
        """
        oldLoss = biggestRecentLoss(self.loss, self.lookback)
        newLoss = float(self.loss[-1])
        fracDiff = 2 * (oldLoss - newLoss)/(oldLoss + newLoss)
        
        if fracDiff < self.ftol:
            
            self.converged = True
        
        
def biggestRecentLoss(losses, memory=3):
    """
    Of the last 'memory' losses, calculcate the largest. 
    """
    memory += 1
    if len(losses) < memory:
        lookback = len(losses)
    else:
        lookback = memory
    oldlosses = losses[-lookback:]
    oldloss = max(oldlosses)
    return oldloss