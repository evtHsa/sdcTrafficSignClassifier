#

import pickle
import pdb
import numpy as np

data_dir = "traffic-signs-data"

def get_pickle_dict(set):
    file = open(format("%s/%s.p" % (data_dir, set)), mode = 'rb')
    d = pickle.load(file)
    return d
                  
def get_id2name_dict():
    # get a dictionary whose keys are the sign id's and whose values are the sign names
    f = open('signnames.csv', 'r')
    id2name_dict = dict(line.strip().split(',') for line in f.readlines()[1:])
    return id2name_dict


def get_sample_signs(X, y, n_classes, name_dict):
    # ouput: a list of tuples where each tuple is (ix, name, image)
    ret = list()
    
    for i in range(n_classes):
        print("do something useful %d" %i)
