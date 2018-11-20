import pickle
import pdb
import numpy as np
import tsc_util as tscu
import matplotlib.pyplot as plt
from textwrap import wrap

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
        return self.sample_sign

    def sample_grid_dims(self):
        n_imgs = len(self.sample_signs)
        rows = int(np.sqrt(n_imgs) - 1)
        cols = np.ceil(n_imgs / rows)
        return rows, cols
    
    def show_sample_signs(self):
        # https://matplotlib.org/gallery/subplots_axes_and_figures/figure_title.html
         #https://stackoverflow.com/questions/10351565/how-do-i-fit-long-title
        img_side = 32 #FIXME: we should calc this in ctor and make as attr
        rows, cols = self.sample_grid_dims()
        font_size = 10 # FIXME:hardcoded badness
        text_width_char = 20
        fig_height = 48
        fig_width = 48
        
        plt.figure(1, figsize=(fig_height, fig_width))
        for i in range(len(self.sample_signs)):
            img = self.sample_signs[i]['img']
            name = self.sample_signs[i]['name']
            plt.subplot(rows, cols, i + 1)
            plt.title("\n".join(wrap("\n%d: %s" % (i + 1, name), text_width_char)),
                      fontsize=font_size)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout(pad=0., w_pad=0., h_pad=1.0)
        plt.show()
        pdb.set_trace()

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
                                          
