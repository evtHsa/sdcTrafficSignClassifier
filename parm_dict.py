#!/usr/bin/env python3

#in its own file so we can evolve it and share it with small unit tests

import cv2

# gpd -> global parm dict
parm_dict ={
    'learning_rate' : .00095,
    'mu' : 0,
    'sigma' : 0.1,
    'EPOCHS' :  128,
    'BATCH_SIZE' : 64,
    'GOOD_ENOUGH' : 0.97
}
