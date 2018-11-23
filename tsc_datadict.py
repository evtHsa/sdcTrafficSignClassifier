import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap

class DataDict:

    def get_pickle_dict(self, set):
        file = open(format("%s/%s.p" % (self.data_dir, set)), mode = 'rb')
        d = pickle.load(file)
        return d
    
    def set_n_classes(self):
        all_labels = self.dd[self.set_names[0]]['y']
        for set_name in self.set_names[1:]:
            np.append(all_labels, self.dd[set_name]['y'])
            self.n_classes = len(np.unique(all_labels))

    def get_dict(self):
        return self.dd
    
    def organize_signs_by_id(self):
        self.signs_by_id = dict()
        dd = self.dd
        for set_name in self.set_names:
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
        # FIXME: this fn still needs help. too many poorly understood plot paramters
        # but it works for nwo
        # https://matplotlib.org/gallery/subplots_axes_and_figures/figure_title.html
        #https://stackoverflow.com/questions/10351565/how-do-i-fit-long-title

        img_side = 32 #FIXME: we should calc this in ctor and make as attr
        rows, cols = self.sample_grid_dims()
        font_size = 10 # FIXME:hardcoded badness
        text_width_char = 22
        fig_height = 24
        fig_width = 24
        plt.figure(1, figsize=(fig_height, fig_width))
        for i in range(len(self.sample_signs)):
            img = self.sample_signs[i]['img']
            name = self.sample_signs[i]['name']
            plt.subplot(rows, cols, i + 1)
            plt.title("\n".join(wrap("\n%d: %s" % (i + 1, name), text_width_char)),
                      fontsize=font_size)
            plt.imshow(img)
            plt.axis('off')
        plt.tight_layout(pad=3., w_pad=1., h_pad=4.0)
        plt.show()
        plt.close()

    def show_distribution(self, set_name):
        plt.bar(range(self.n_classes), [len(s) for s in self.signs_by_id[set_name]],
                align='center')
        plt.title("Distribution(%s)" % set_name)
        plt.xticks(range(0, self.n_classes, 5))
        plt.show()
        plt.close()
        
    def show_distributions(self):
        for set_name in self.set_names:
            self.show_distribution(set_name)

    def normalize_images_of_set(self, set_name):
        f = lambda img: (img - 128.) / 128.
        self.dd[set_name]['X'] = np.array([f(img)
                                  for img in self.dd[set_name]['X']])
        
    def normalize_images(self):
        # do this after displaying the sample signs
        for set_name in self.set_names:
            self.normalize_images_of_set(set_name)

    def preprocess_images(self):
        self.normalize_images()
        

    def summarize(self):
        print("loaded from ", self.load_type)
        for set in self.set_names:
            print("\t" + set)
            print("\t\tshape(features) = " + str(self.get_vbl(set, 'X').shape ))
            print("\t\tshape(labels) = " + str(self.get_vbl(set, 'y').shape ))

    def load_from_image_dir(self):
        assert("needs code" == None)
        
    def load_from_pickle(self):
        for set_name in self.set_names:
            pd = self.get_pickle_dict(set_name)
            assert(len(pd['features']) == len(pd['labels']))
            self.dd[set_name] = { 'X' : pd['features'], 'y' : pd['labels']}
            
    def __init__(self, set_names, load_type, data_dir, show_sample=False,
                 show_distrib=False, summarize=False, do_pre_pro=False):
        
        assert(set_names != None)
        assert(type(set_names) is list)
        
        self.set_names = set_names
        self.data_dir = data_dir
        self.dd = dict()

        self.load_type = load_type
        assert(self.load_type == 'pickle' or self.load_type == 'image_dir')
        self.load_fn_dict = {
            'pickle' : self.load_from_pickle,
            'image_dir' : self.load_from_image_dir
            }
        self.load_fn_dict[self.load_type]()
        
        self.set_n_classes()
        self.organize_signs_by_id()
        self.set_id2name_map()
        self.select_sample_signs()
        if show_sample:
            self.show_sample_signs()
        if show_distrib:
            self.show_distributions()
        if do_pre_pro:
            self.preprocess_images()
        if summarize:
            self.summarize()
                                          
