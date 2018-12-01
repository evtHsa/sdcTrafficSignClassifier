#!/usr/bin/env python3
#

import sys

# unit tests are usually run from their own dir or parent dir
# unit tests care only of standard stuff and parent dir
sys.path.append(".") 
sys.path.append("..") 


# debug stuff
import pdb

##cell
# support code
import tsc_datadict as tsc_dd
import parm_dict
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np

##cell
DD = tsc_dd.DataDict([ 'train', 'valid', 'test'], 'pickle', 'traffic-signs-data')

#load hyperparms
pd = parm_dict.parm_dict # reduce typing

def ldhp(key):
    ret = pd[key]
    print("%s -> %s" % (key, ret))
    return ret

rate = ldhp('learning_rate')
mu = ldhp('mu')
sigma = ldhp('sigma')
EPOCHS = ldhp('EPOCHS')
BATCH_SIZE=ldhp('BATCH_SIZE')
GOOD_ENOUGH=ldhp('GOOD_ENOUGH')

##cell@
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

n_train = DD.get_vbl('train', 'X').shape[0]
n_validation = DD.get_vbl('valid', 'X').shape[0]
n_test = DD.get_vbl('test', 'X').shape[0]
_X = DD.get_vbl('valid', 'X')
image_shape = "(" + str(_X.shape[1]) + ", " + str(_X.shape[2]) + ',' + str(_X.shape[3]) + ")"
n_classes = DD.n_classes

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

##cell
DD.show_sample_signs()
DD.show_distributions()

##cell
##cell
##cell
##cell
##cell
##cell
##cell
##cell
##cell
##cell
