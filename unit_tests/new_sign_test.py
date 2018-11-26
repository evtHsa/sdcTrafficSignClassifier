#!/usr/bin/env python3
#

import sys

# unit tests are usually run from their own dir or parent dir
# unit tests care only of standard stuff and parent dir
sys.path.append(".") 
sys.path.append("..") 


# debug stuff
import pdb

# support code
import tsc_datadict as tsc_dd
import tensorflow as tf

# https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/


def main():
    DD = tsc_dd.DataDict([ 'test'], 'image_dir', 'found_signs')
    DD.summarize()
    DD.show_sample_signs()
    X = DD.get_vbl('test', 'X')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('lenet.meta')
        saver.restore(sess, "./lenet")
        pdb.set_trace()
        predictions = sess.run(prediction, feed_dict={x: X, keep_prob: 1.0})
        pdb.set_trace()

main()
print("done")
print("FIXME: what happened to the histogram")
