# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:59:17 2016

@author: ibackus
"""
import csv
import os

_directory = os.path.dirname(os.path.realpath(__file__))

datafile = os.path.join(_directory, 'JSE_OkCupid', 'profiles.csv')

def loadprofiles(filename=datafile):
    
    # Load data
    with open(filename) as f:
        reader = csv.reader(f)
        data = [row for row in reader]
        header = data.pop(0)
    
    # Parse data
        
    return header, data
    
    