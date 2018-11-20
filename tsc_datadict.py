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
            self.dd['n_classes'] = len(np.unique(all_labels))
        
    def get_dict(self):
        return self.dd

    def __init__(self):
        self.dd = dict()
        for set_name in self.get_set_names():
            pd = tscu.get_pickle_dict(set_name)
            self.dd[set_name] = { 'X' : pd['features'], 'y' : pd['labels']}
        self.set_n_classes()
