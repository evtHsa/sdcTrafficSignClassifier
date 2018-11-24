#!/usr/bin/env python3
#

import sys

# unit tests are usually run from their own dir or parent dir
# unit tests care only of standard stuff and parent dir
sys.path.append(".") 
sys.path.append("..") 


# debug stuff
import pdb

# support code
import tsc_datadict as tsc_dd
from sklearn.utils import shuffle

#DD = tsc_dd.DataDict([ 'train', 'valid', 'test'], 'pickle', 'traffic-signs-data')
#DD.summarize()
#DD.show_sample_signs()
#print("X is a ", type(DD.get_vbl('train', 'X').shape[0]))

#n_train = DD.get_vbl('train', 'X').shape[0]

DD = tsc_dd.DataDict([ 'test'], 'image_dir', 'found_signs')
DD.summarize()
DD.show_sample_signs()
print("done")
