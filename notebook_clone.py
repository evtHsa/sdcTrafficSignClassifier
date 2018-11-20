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
DD = tsc_dd.DataDict()
dd = DD.get_dict()
print("FIXME: the above is a bit stupid")

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np

# TODO: Number of training examples
n_train = dd['train']['X'].shape[0]

# TODO: Number of validation examples
n_validation = dd['valid']['X'].shape[0]

# TODO: Number of testing examples.
n_test = dd['test']['X'].shape[0]

# 2DO: What's the shape of an traffic sign image?
image_shape = "(" + str(dd['train']['X'].shape[1]) + "," + str(dd['train']['X'].shape[2]) + ")"

# 2DO: How many unique classes/labels there are in the dataset.
# now part of dd

print("shape(features) = " + str(dd['train']['X'].shape ))
print("shape(labels) = " + str(dd['train']['y'].shape ))
print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", DD.n_classes)

sn_dict = tscu.get_id2name_dict()

signs_by_set_by_id = DD.signs_by_id

assert("FIXME:change the signature" == None)

sample_signs = get_sample_signs(dd['train']['X'], dd['train']['y'], n_classes, sn_dict,
                                signs_by_set_by_id['train'])

pdb.set_trace()
