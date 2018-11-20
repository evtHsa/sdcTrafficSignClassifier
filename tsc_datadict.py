import pickle
import pdb
import numpy as np
import tsc_util as tscu

class DataDict:
    def get_set_names(self):
        return [ 'train', 'valid', 'test']

    def set_n_classes(self):
        set_names = self.get_set_names()
        all_labels = self.dd[set_names[0]]['y']
        for set_name in set_names[1:]:
            np.append(all_labels, self.dd[set_name]['y'])
            self.n_classes = len(np.unique(all_labels))
        
    def get_dict(self):
        return self.dd
    
    def organize_signs_by_id(self):
        self.signs_by_id = dict()
        set_names = self.get_set_names()
        dd = self.dd
        for set_name in set_names:
            labels = dd[set_name]['y']
            self.signs_by_id[set_name] = [[ix for ix, id in
                                           enumerate(labels) if id == i] for i in
                                          range(self.n_classes)]
    def get_vbl(self, set_name, vbl_name):
        return self.dd[set_name][vbl_name]
    
    def __init__(self):
        self.dd = dict()
        for set_name in self.get_set_names():
            pd = tscu.get_pickle_dict(set_name)
            self.dd[set_name] = { 'X' : pd['features'], 'y' : pd['labels']}
        self.set_n_classes()
        self.organize_signs_by_id()
                                          
