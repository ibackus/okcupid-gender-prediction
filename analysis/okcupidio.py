# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:59:17 2016

@author: ibackus
"""
import csv
import os
import json
import numpy as np
import cPickle as pickle
from datetime import datetime
import time
import utils

def converter(data):
    """
    Assume setupInfo has been called and perform basic converting
    """
    for dataname, x in zip(data.keys(), data.values()):
        
        dtype = x['dtype']
        xdata = x['str_data']
        
        print dataname, dtype
        
        if dtype == 'int':
            d = str2int(xdata)
        elif dtype == 'date':
            d = str2date(xdata)
        elif dtype == 'list':
            d, opts = filterList(xdata)
            x['options'] = opts
        elif dtype == 'ranked':
            d = filterRanked(xdata, x['ranking'])
        elif dtype == 'multiple choice':
            d = multiChoiceFilter(xdata, x['options'])
        else:
            d = xdata
        
        x['data'] = d
    
def categorize(data):
    """
    """
    for field in data.keys():
        
        data[field]['dtype'] = categoryTypes.get(field, None)
        
def processData(data):
    """
    Runs initial processing of data loaded via loadData()
    """
    # Tell which category each data field is ('int', 'date', etc..)
    categorize(data)
    
    # Setup the choices for the data
    for field, val in data.iteritems():
        
        if val['dtype'] in ('multiple choice', 'other', 'ranked'):
            
            setupChoices(val)
            
        if val['dtype'] == 'ranked':
            
            val['ranking'] = rankings.get(field, None)
            
    # Do pre-parsing for some fields
    parseOffspring(data)
    parseSign(data)
    parseReligion(data)
    parsePets(data)
    # Perform basic converting for availble data types
    converter(data)
    # Do post-parsing
    parseSpeaks(data)
    parseEthnicity(data)
    
def loadData(filename=None):
    """
    Loads the profile data from the .csv file.
    
    '' and 'NA' are replace with None
    """
    if filename is None:
        filename = datafile
    # Load data
    with open(filename) as f:
        reader = csv.reader(f)
        data = [row for row in reader]
        header = data.pop(0)
    
    # save data as vectors for each profile attribute
    dataDict = {}
    
    for i, category in enumerate(header):
        
        column = [d[i] for d in data]                
        dataDict[category] = {'str_data': column}
    
    ignoreFilter(dataDict)
    return dataDict

# ---------------------------
# FEATURE BUILDING
# ---------------------------
def buildFeatures(data):
    """
    A technique for building features.  
    NOTE! This will alter data in-place.  All multiple choice fields will 
    be converted to binary arrays
    
    At the end of this:
        
        multiple choice options are converted to binary arrays
        dates, ints are left as-is as arrays
        ranked datas are treated as numbers 
    """
    # Now concatenate things into a single array
    featureNames = []
    features = []
    for field, x in data.iteritems():
        
        # Convert multiple choice things to binary arrays
        if x['dtype'] == 'multiple choice':
            
            
            x['data']  = multi2binary(x['data'])
            x['dtype'] = 'binary array'
            
        if x['dtype'] == 'binary array':
            
            if len(x['options']) == 2:
                # There are only two options
                name = field + '_' + x['options'][1]
                featureNames.append(name)
            else:
                # Assume there are N data fields (one for every option)
                # That is to say, binary arrays are just yes/no questions, eg
                # are you aquarius?
                names = [field + '_' + opt for opt in x['options']]
                featureNames.extend(names)
                
            features.append(x['data'])
            
        elif x['dtype'] in ('date', 'int', 'ranked'):
            
            featureNames.append(field)
            xarray = np.asarray(x['data'])
            xarray = xarray.reshape([len(xarray), 1])
            features.append(xarray)
        
    # Now cast as arrays
    features = np.concatenate(features, axis=1)
    featureNames = np.asarray(featureNames)
    return features, featureNames
    
def multi2binary(x):
    """
    x should be ints 0, 1, 2, ... nClass-1 
    -1 in the input is treated as a missing value and is replaced with -1
    
    Returns a numpy array shape (n x nClass) if nClass != 2
    
    For nClass == 2, just return the data unaltered
    """
    x = np.asarray(x)
    nClass = x.max() + 1
    
    if nClass == 2:
        
        return utils.columnVector(x)
        
    xbin = np.zeros([len(x), nClass])
    
    for i in range(nClass):
        
        mask = (x == i)
        xbin[mask, i] = 1
    
    mask = (x == -1)
    xbin[mask, :] = -1
    return xbin
# ---------------------------
# field specific parsers
#
# ---------------------------
def parseEthnicity(data):
    """
    VERY simple parser.  Should be done AFTER splitting (after doing converter)
    
    Thus input vals of the data (a list of lists) are 0, 1, ... nClass-1 and
    -1 for missing data
    
    In the output, missing data is marked as rows equal to -1
    """
    x = data['ethnicity']['data']
    # Get number of options
    nOpts = len(data['ethnicity']['options'])
    # Convert to a binary array
    xbin = np.zeros([len(x), nOpts])
    for i, line in enumerate(x):
        for val in line:
            if val == -1:
                xbin[i, :] = -1
            else:
                xbin[i, val] = 1
    data['ethnicity']['data'] = xbin
    data['ethnicity']['dtype'] = 'binary array'
    
def parseSpeaks(data):
    """
    A VERY simple parser.  treat speaking a language poorly as not speaking
    it, otherwise treat it is being spoken (ie don't differentiate between 
    okay and fluently)
    
    This should be applied AFTER splitting 'speaks'
    
    Then this just splits this into a binary array, 0 means you speak the
    language, 1 means you don't
    
    While we're at it, I'll also keep track of how many languages you speak
    """
    x = data['speaks']['str_data']
    x = splitList(x)
    # Remove tags and replace speaking Poorly with not speaking at all
    oktags = [' (fluently)', ' (okay)']
    badtags = ['(poorly)']
    for line in x:
        for i, entry in enumerate(line):
            if any(tag in entry for tag in badtags):
                line[i] = ''
            else:
                for tag in oktags:
                    if tag in entry:
                        line[i] = entry.replace(tag, '')
    # Flatten x and get a set of options (ignore nones)
    languages = list(set(item for sublist in x for item in sublist if item not \
                         in noneStrings))
    languages.sort()
    data['speaks']['options'] = languages
    nlang = len(languages)
    xbin = np.zeros([len(x), nlang])
    for i, line in enumerate(x):
        for language in line:
            if language not in noneStrings:
                ind = languages.index(language)
                xbin[i, ind] = 1
    nspoken = xbin.sum(1)
    data['speaks']['data'] = xbin
    data['speaks']['dtype'] = 'binary array'
    data['num_languages'] = {'dtype': 'int', 'data': nspoken}
    
def parseReligion(data):
    x = data['religion']
    
    religion = []
    serious = []
    
    for line in x['str_data']:
        
        r, s = _parseReligion(line)
        religion.append(r)
        serious.append(s)
    
    data['religion'] = {'dtype': 'multiple choice', 'str_data': religion}
    data['religion_importance'] = {'dtype': 'ranked', 'str_data': serious}
    setupChoices(data['religion'])
    setupChoices(data['religion_importance'])
    ranking = ['laughing about it',
               'not too serious about it',
               'somewhat serious about it',
               'very serious about it']
    data['religion_importance']['ranking'] = ranking
    
def _parseReligion(line):
    
    if line in noneStrings:
        
        return '', ''
    
    words = line.split(' ')
    religion = words[0]
    
    if len(words) > 1 and (words[1] in ('and', 'but')):
        
        serious = ' '.join(words[2:])
        
    else:
        
        serious = ''
        
    return religion, serious
    
    
def parsePets(data):
    """
    Assume having a cat or a dog means you like it
    """
    x = data['pets']
    results = {'has_cats': [], 'has_dogs': [], 'likes_cats': [], 'likes_dogs': []}
    
    for line in x['str_data']:
        
        r = _parsePets(line)
        for k, v in r.iteritems():
            results[k].append(v)
            
    for key, val in results.iteritems():
        
        data[key] = {'dtype': 'int', 'str_data': val}
    
    data.pop('pets', None)
    
    
def _parsePets(line):
    """
    """
    likes_cats = -1
    likes_dogs = -1
    has_cats = -1
    has_dogs = -1
    
    if 'has cats' in line:
        likes_cats = 1
        has_cats = 1
    else:
        has_cats = 0
    
    if 'has dogs' in line:
        likes_dogs = 1
        has_dogs = 1
    else:
        has_dogs = 0
    
    if 'dislikes cats' in line:
        likes_cats = 0
    elif 'likes cats' in line:
        likes_cats = 1
    
    if 'dislikes dogs' in line:
        likes_dogs = 0
    elif 'likes dogs' in line:
        likes_dogs = 1
    
    return {'has_cats': has_cats, 'has_dogs': has_dogs, \
    'likes_cats': likes_cats, 'likes_dogs': likes_dogs}
    
    
def parseSign(data):
    
    x = data['sign']
    signs = []
    sign_matters = []
    
    for line in x['str_data']:
        
        s, m = _parseSign(line)
        signs.append(s)
        sign_matters.append(m)
        
    data['sign'] = {'str_data': signs, 'dtype': 'multiple choice'}
    data['sign_matters'] = {'str_data': sign_matters, 'dtype': 'multiple choice'}
    setupChoices(data['sign'])
    setupChoices(data['sign_matters'])
    
def _parseSign(line):
    """
    Parse one line of the 'sign' field data
    """
    if line in noneStrings:
        return '', ''        
    if 'and' in line:
        line = line.split('and')
    elif 'but' in line:
        line = line.split('but')
    else:
        line = [line, '']
    sign = line[0].strip()
    sign_matters = line[1].strip()
    
    return sign, sign_matters
        
    
    
def parseOffspring(data):
    """
    Parse the 'offspring' field
    
    Removes the 'offspring' field
    """
    key = 'offspring'
    x = data[key]
    
    # Parse
    haskids = []
    wantsmore = []
    for line in x['str_data']:
        has, wants = _parseOffspring(line)
        haskids.append(has)
        wantsmore.append(wants)
    
    # Save/cleanup
    data['has_kids'] = {'str_data': haskids, 'dtype': 'int'}
    data['wants_more_kids'] = {'str_data': wantsmore, 'dtype': 'multiple choice'}
    setupChoices(data['wants_more_kids'])
    data.pop(key, None)
        
def _parseOffspring(line):
    """
    Parse one line of the offspring data field
    """
    if line in noneStrings:
        haskids = ''
        wantsmore = ''
    else:
        haskids = str(int('has a kid' in line))
        if ('doesn&rsquo;t want any' or 'doesn&rsquo;t want more') in line:
            wantsmore = 'no'
        elif 'might want' in line:
            wantsmore = 'maybe'
        elif ('wants them' or 'wants more') in line:
            wantsmore = 'yes'
        else:
            wantsmore = ''
            
    return haskids, wantsmore
# ---------------------------
# FILTERS
# ---------------------------
def filterList(x, delimiter=','):
    """
    Process data that is a list of options (comma separated)
    """
    x2 = splitList(x, delimiter)
    # Here's a bit of pythonic magic.  What's going on here is we're flattening
    # x (a list of lists) and simultaneously making a set out of the entries
    options = set(item for sublist in x2 for item in sublist)
    # Ignore nonestring options
    options = [opt for opt in options if opt not in noneStrings]
    # Now make a sorted list
    options = list(options)
    options.sort()
    # Now replace occurences of the options with their number value
    for xi in x2:
        for i, val in enumerate(xi):
            if val in noneStrings:
                xi[i] = -1
            else:
                xi[i] = options.index(val)
    return x2, options
    
def multiChoiceFilter(x, options):
    """
    """
    if isinstance(x, list):
        
        return [multiChoiceFilter(d, options) for d in x]
        
    if x in noneStrings:
        
        return -1
        
    else:
        
        return options.index(x)
    
def ignoreFilter(dataDict):
    """
    Initial inplace ignoring of unwanted keys
    """
    # Filter out columns to ignore
    for ignoreKey in ignore:
        
        if ignoreKey in dataDict:
            
            dataDict.pop(ignoreKey, None)
            
def str2date(date, fmt='%Y-%m-%d-%H-%M'):
    """
    Convert the string date (or list of strings) to unix time using fmt 
    (see datetime.datetime.strptime)
    """
    if isinstance(date, list):
            
        return [str2date(d) for d in date]
    
    if date in noneStrings:
        
        return -1
        
    dateobj = datetime.strptime(date, fmt)
    unixtime = time.mktime(dateobj.timetuple())
    return unixtime
    
def str2int(x):
    """
    Convert a str or list of strings to int(s)
    """
    if isinstance(x, list):
        return [str2int(a) for a in x]
        
    if x in noneStrings:
        
        x = -1
        
    else:
        
        x = int(x)
        
    return x

def splitList(x, delimiter=','):
    """
    """
    if isinstance(x, list):
        
        return [splitList(a, delimiter) for a in x]
        
    xSplit = x.split(delimiter)
    for i, val in enumerate(xSplit):
        
        xSplit[i] = val.strip()
            
    return xSplit

def filterRanked(col, ranking):
    """
    replace col values with their rank number for a given ranked category
    
    Empty strings are replaced with -1
    
    ranking is a list of the options, ranked in ascending order
    """
    data = []
    # Filter ranked values
    for i in range(len(col)):
        
        x = col[i]
        if x in noneStrings:            
            newval = -1            
        else:
            newval = ranking.index(x)
        data.append(newval)
        
    return data

def rankNames(col, category):
    """
    Return the strings associated with rankings for a given ranked category
    """
    data = []
    # Filter ranked values
    if categoryTypes[category] != 'ranked':
        
        raise ValueError, 'Category {0} is not ranked'.format(category)
        
    ranking = rankings[category]
    for x in col:
        
        if x == -1:
            
            data.append('')
            
        else:
            
            data.append(ranking[x])
        
    return data
    
def printProfile(prof, header):
    """
    """
    rows = [list(row) for row in zip(header, prof)]
    col_width = max(len(h) for h, p in rows) + 2
    for row in rows:
        
        # Indent after any newlines
        row[1] = row[1].replace('\n', '\n' + ' '*col_width)
        # Print with column formatting
        print "".join(word.ljust(col_width) for word in row)
    
    return
    
def setupChoices(x):
    """
    x should be a dict with the 'str_data' field
    
    x['options'] = sorted set of things in x['str_data']
    """
    data = x['str_data']
    options = list(set(data))
    options = [opt for opt in options if opt not in noneStrings]
    options.sort()
    x['options'] = options
    
_directory = os.path.dirname(os.path.realpath(__file__))

# Set up available choices
_choicesFile = os.path.join(_directory, 'data', 'choices.json')


datafile = os.path.join(_directory, 'JSE_OkCupid', 'profiles.csv')

# For the prefiltering stage, some things are multiple choice + modifier
modifiers = {}
modifiers['sign'] = ['it matters a lot', 
                    'it&rsquo;s fun to think about', 
                    'but it doesn&rsquo;t matter']
modifiers['diet'] = ['mostly', 'strictly']
modifiers['education'] = ['dropped out of', 'graduated from', 'working on']
              
# Also, lets just ignore certain keys for now
ignore = ['location']

# Define the response types for the categories
categoryTypes = {'age': 'int',
'body_type': 'multiple choice',
'diet': 'multiple choice',
'drinks': 'ranked',
'drugs': 'ranked',
'education': 'multiple choice',
'essay0': 'short response',
'essay1': 'short response',
'essay2': 'short response',
'essay3': 'short response',
'essay4': 'short response',
'essay5': 'short response',
'essay6': 'short response',
'essay7': 'short response',
'essay8': 'short response',
'essay9': 'short response',
'ethnicity': 'list',
'height': 'int',
'income': 'int',
'job': 'multiple choice',
'last_online': 'date',
'location': 'multiple choice',
'offspring': 'other',
'orientation': 'multiple choice',
'pets': 'other',
'religion': 'other',
'sex': 'multiple choice',
'sign': 'other',
'smokes': 'multiple choice',
'speaks': 'list',
'status': 'multiple choice'}
availTypes = list(set(categoryTypes.values()))
availCategories = categoryTypes.keys()

categoryMembers = {}
for t in availTypes:
    
    categoryMembers[t] = [k for k, v in categoryTypes.iteritems() if v == t]

ranked = [k for k, v in categoryTypes.iteritems() if v == 'ranked']
other = [k for k, v in categoryTypes.iteritems() if v == 'other']
multiple_choice = [k for k, v in categoryTypes.iteritems() if v == 'multiple choice']

# Define rankings for 'ranked' categories
rankings = {}
rankings['drugs'] = ['never', 'sometimes', 'often']
rankings['drinks'] = ['not at all', 'rarely', 'socially', 'often', \
    'very often', 'desperately']
    
noneStrings = ('', 'NA', None)