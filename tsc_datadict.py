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
    
    def get_dict(self):
        return self.dd
    
    def organize_signs_by_id(self):
        self.signs_by_id = dict()
        d_ = dict()
        l_ = list()
        for set_name in self.set_names:
            labels = self.dd[set_name]['y']
            ix_class_list = enumerate(labels)
            for label in np.unique(labels):
                d_[label] = list()
            for ix, class_ix in ix_class_list:
                d_[class_ix].append(ix)
            for ix in sorted([int(ix) for ix in d_.keys()]):
                l_.append(d_[str(ix)])
            self.signs_by_id[set_name] = d_
            
    def get_vbl(self, set_name, vbl_name):
        return self.dd[set_name][vbl_name]

    def process_class_names(self):
        f = open('signnames.csv', 'r')
        lines = f.readlines()[1:]
        self.id2name_dict = dict(line.strip().split(',')
                                 for line in lines)
        self.n_classes = len(np.unique(self.id2name_dict.keys())[0])

    def dump_sample_signs(self):
        print("=== dump_sample_signs ===")
        for set_name in self.sample_signs.keys():
            print("\t", set_name)
            dict_list = self.sample_signs[set_name]
            for _dict in dict_list:
                print("\t\tix = %d, class = %d, name = %s" % (_dict['ix'],
                                                              _dict['sign_class'], _dict['name']))

    def select_sample_signs_1st_of_class_by_set(self, set_name_list):
        for set_name in set_name_list:
            _list = self.sample_signs[set_name]
            for class_ix in self.signs_by_id[set_name].keys():
                ix = self.signs_by_id[set_name][class_ix][0]
                _list.append({ 'ix' : ix, 'sign_class' : class_ix,
                               'name' : self.id2name_dict[str(class_ix)]})
        self.dump_sample_signs()
            
    def select_sample_signs_all_per_class(self, set_name_list):
        print("FIXME: select_sample_signs_all_per_class")

        for set_name in set_name_list:
            _list = self.sample_signs[set_name]
            for class_ix_s in self.signs_by_id[set_name].keys():
                for img_ix in self.signs_by_id[set_name][class_ix_s]:
                    _list.append({ 'ix': img_ix, 'sign_class' : i,
                                   'name' : self.id2name_dict[str(i)]})
        self.dump_sample_signs()
        pdb.set_trace()
        print("FIXME: check sample_signs")

    def get_sample_signs(self):
        return self.sample_signs

    def sample_grid_dims(self):
        # FIXME: analogize what I did for lane finding, fixed grid, repeat until all dpy'ed
        # FIXME: the displayer should be a much simpler class than for lane finding
        rows = 9
        cols = 5
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

        for set_name in self.sample_set_list:
            X = self.get_vbl(set_name, 'X')
            i = 0
            for ssd in self.sample_signs[set_name]: #ssd -> sample sign dict
                sign_class      = ssd['sign_class']
                img   = X[ssd['ix']]
                name = ssd['name']
                print("FIXME: ix = %d, class = %d, name = %s" %(ssd['ix'],
                                                                sign_class, name))
                plt.subplot(rows, cols, i + 1)
                plt.title("\n".join(wrap("\n%d: %s" % (sign_class + 1, name),
                                         text_width_char)),
                          fontsize=font_size)
                plt.imshow(img)
                plt.axis('off')
                i += 1
            pdb.set_trace()
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
        list_of_file_class_pair_lists = [line.strip().split(',') for line in lines]
        class2img_name_dict = dict()
        for filename, class_ix_s in list_of_file_class_pair_lists:
            class_ix_s = class_ix_s.strip() # took a while to see _this_ problem
            if not class_ix_s in class2img_name_dict:
                class2img_name_dict[class_ix_s] = [filename]
            else:
                class2img_name_dict[class_ix_s].append(filename)
        return class2img_name_dict

    def load_csv_file(self, set_name):
            cind = self.parse_img2class_csv(set_name)
            #cind -> class 2 image name dict
            X = list()
            y = list()
            for class_str in [str(ix) for ix in sorted([ int(x) for x in cind.keys()])]:
                for img_name in cind[class_str]:
                    img_path = self.data_dir + "/" + img_name
                    X.append(plt.imread(img_path))
                    y.append(class_str)
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
        self.sample_signs = dict()
        for set_name in set_names:
            self.sample_signs[set_name] = list()

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

        # save set_name_list since show_sample_signs may be called much later
        self.sample_set_list =  self.fn_dict_dict[self.load_type]['sample_set_list']

        self.fn_dict_dict[self.load_type]['load_fn']()
        self.process_class_names() # sets n_classes, id2name_dict
        self.organize_signs_by_id()
        sample_select_fn = self.fn_dict_dict[self.load_type]['sample_select']
        sample_select_fn(self.fn_dict_dict[self.load_type]['sample_set_list'])
        
        if show_sample:
            self.show_sample_signs()
        if show_distrib:
            self.show_distributions()
        if do_pre_pro:
            self.preprocess_images()
        if summarize:
            self.summarize()
                                          
