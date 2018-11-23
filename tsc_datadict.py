import pickle
import pdb
import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
import cv2

class DataDict:

    def get_pickle_dict(self, set):
        file = open(format("%s/%s.p" % (self.data_dir, set)), mode = 'rb')
        d = pickle.load(file)
        return d
    
    def set_n_classes(self):
        all_labels = np.array([])
        for set_name in self.set_names:
            all_labels = np.append(all_labels, self.dd[set_name]['y'])
        self.n_classes = len(np.unique(all_labels))

    def get_dict(self):
        return self.dd
    
    def organize_signs_by_id(self):
        pdb.set_trace()
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

    def select_sample_signs_1st_of_class_by_set(self, set_name_list):
        for set_name in set_name_list:
            for i in range(self.n_classes):
                X = self.get_vbl(set_name, 'X')
                ix = self.signs_by_id[set_name][i][0]
                img =X[ix]
                self.sample_signs.append({ 'sign_class' : i, 'img' : img,
                                           'name' : self.id2name_dict[str(i)]})
            
    def select_sample_signs_all_per_class(self, set_name_list):
        pdb.set_trace()
        for set_name in set_name_list:
            for i in range(self.n_classes):
                X = self.get_vbl(set_name, 'X')
                for ix in self.signs_by_id[set_name][i]:
                    img =X[ix]
                    self.sample_signs.append({ 'sign_class' : i, 'img' : img,
                                               'name' : self.id2name_dict[str(i)]})

    def get_sample_signs(self):
        return self.sample_signs

    def sample_grid_dims(self):
        pdb.set_trace()
        n_imgs = len(self.sample_signs)
        rows = int(np.sqrt(n_imgs) - 1)
        cols = np.ceil(n_imgs / rows)
        return rows, cols
    
    def show_sample_signs(self):
        # FIXME: this fn still needs help. too many poorly understood plot paramters
        # but it works for nwo
        # https://matplotlib.org/gallery/subplots_axes_and_figures/figure_title.html
        #https://stackoverflow.com/questions/10351565/how-do-i-fit-long-title

        duh = True
        img_side = 32 #FIXME: we should calc this in ctor and make as attr
        rows, cols = self.sample_grid_dims()
        print("FIXME:rows = %d, cols = %d" % (rows, cols))
        font_size = 10 # FIXME:hardcoded badness
        text_width_char = 22
        fig_height = 24
        fig_width = 24
        plt.figure(1, figsize=(fig_height, fig_width))
        i = 0
        for sample_sign_dict in self.sample_signs:
            sign_class      = sample_sign_dict['sign_class']
            img   = sample_sign_dict['img']
            if duh:
                print("type(img) = ", type(img))
                duh = False
            name = sample_sign_dict['name']
            plt.subplot(rows, cols, i + 1)
            plt.title("\n".join(wrap("\n%d: %s" % (sign_class + 1, name), text_width_char)),
                      fontsize=font_size)
            plt.imshow(img)
            plt.axis('off')
            i += 1
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
        print("\t\tn_classes = ", self.n_classes)

    def parse_img2class_csv(self, set_name):
        fname = "%s/%s.csv" % (self.data_dir, set_name)
        f = open(fname, 'r')
        lines  = f.readlines()[1:]
        ret = dict(line.strip().split(',') for line in lines)
        return ret

    def load_csv_file(self, set_name):
            print("load_csv_file: ", set_name)
            csv_dict = self.parse_img2class_csv(set_name)
            X = list()
            y = list()
            
            for fname in csv_dict.keys():
                img_path = self.data_dir + "/" + fname
                X.append(plt.imread(img_path))
                y.append(csv_dict[fname])
            return np.array(X), np.array(y)

    def load_from_image_dir(self):
        for set_name in self.set_names:
            image_array, class_array = self.load_csv_file(set_name)
        self.dd[set_name] = { 'X' : image_array, 'y' : class_array}
        
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
        self.sample_signs = list()

        self.load_type = load_type
        assert(self.load_type == 'pickle' or self.load_type == 'image_dir')
        self.fn_dict_dict = {
            'pickle' : {
                'load_fn' : self.load_from_pickle,
                'sample_set_list' : ['train'],
                'sample_select' : self.select_sample_signs_1st_of_class_by_set
            },
            'image_dir' : {
                'load_fn' : self.load_from_image_dir,
                'sample_set_list' : ['test'],
                'sample_select' : self.select_sample_signs_all_per_class}
            }

        self.fn_dict_dict[self.load_type]['load_fn']()
        
        self.set_n_classes()
        self.organize_signs_by_id()
        self.set_id2name_map()
        self.fn_dict_dict[self.load_type]['sample_select'](
            self.fn_dict_dict[self.load_type]['sample_set_list'])
        if show_sample:
            self.show_sample_signs()
        if show_distrib:
            self.show_distributions()
        if do_pre_pro:
            self.preprocess_images()
        if summarize:
            self.summarize()
                                          
