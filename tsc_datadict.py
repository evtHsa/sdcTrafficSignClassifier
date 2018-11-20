import pickle
import pdb
import numpy as np
import tsc_util as tscu

class DataDict:

    def get_pickle_dict(self, set):
        file = open(format("%s/%s.p" % (self.data_dir, set)), mode = 'rb')
        d = pickle.load(file)
        return d
    
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

    def set_id2name_map(self):
        f = open('signnames.csv', 'r')
        self.id2name_dict = dict(line.strip().split(',')
                                 for line in f.readlines()[1:])

    def select_sample_signs(self):
        # only operate on training set
        self.sample_signs = [None] * self.n_classes
        for i in range(self.n_classes):
            X = self.get_vbl('train', 'X')
            # for now just select 1st image of every class
            ix = self.signs_by_id['train'][i][0]
            img =X[ix]
            self.sample_signs[i] = { 'img' : img,
                                     'name' : self.id2name_dict[str(i)]}

    def get_sample_signs(self):
        return self.sample_signs

    def __init__(self):
        self.data_dir = "traffic-signs-data"
        self.dd = dict()
        for set_name in self.get_set_names():
            pd = self.get_pickle_dict(set_name)
            self.dd[set_name] = { 'X' : pd['features'], 'y' : pd['labels']}
        self.set_n_classes()
        self.organize_signs_by_id()
        self.set_id2name_map()
        self.select_sample_signs()
                                          
