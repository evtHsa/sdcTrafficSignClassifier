#!/usr/bin/env python3
#

# debug stuff
import pdb

# support code
import tsc_util as tscu
import tsc_datadict as tsc_dd

#step 0
# Load pickled data
import pickle
    
# TODO: Fill this in based on where you saved the training and testing data
DD = tsc_dd.DataDict(show_sample=True, show_distrib=True)

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np

# TODO: Number of training examples
n_train = DD.get_vbl('train', 'X').shape[0]

n_validation = DD.get_vbl('valid', 'X').shape[0]

# TODO: Number of testing examples.
n_test = DD.get_vbl('test', 'X').shape[0]

# 2DO: What's the shape of an traffic sign image?
_X = DD.get_vbl('valid', 'X')
image_shape = "(" + str(_X.shape[1]) + "," + str(_X.shape[2]) + ")"

# 2DO: How many unique classes/labels there are in the dataset.
# now part of dd

print("shape(features) = " + str(DD.get_vbl('train', 'X').shape ))
print("shape(labels) = " + str(DD.get_vbl('train', 'y').shape ))
print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", DD.n_classes)

pdb.set_trace()
