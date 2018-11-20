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

def organize_signs_by_id(labels_name_pair_list, n_classes):
    # input: a list of pairs (label_nd_llist, string_name_for_list)
    # ouput: a dict with keys from the name part of the label_name_pair
    #            for each key, the value is a list of n_classes in length where the list elements
    #            are lists of the indexes of signs with id = list position
    #    perhaps more understandably
    #    { 'train' : [[list of indices where sign_id is 0], [list of indices where sign_id is 1],
    #                    ..., [list of indices where sign_id is n_classes]]
    #      'eval' : [likewise]
    #      'test' : [likewise]}
    lnpl = labels_name_pair_list
    
    assert(lnpl != None)

    d = dict()
    for labels, labels_name in lnpl:
        d[labels_name] = [[ix for ix, id in enumerate(labels) if id == i] for i in range(n_classes)]
    return d

def get_sample_signs(X, y, n_classes, name_dict):
    # ouput: a list of tuples where each tuple is (ix, name, image)
    ret = list()
    
    for i in range(n_classes):
        print("do something useful %d" %i)
