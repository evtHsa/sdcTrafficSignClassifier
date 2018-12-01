#!/usr/bin/env python3
#

import sys

# unit tests are usually run from their own dir or parent dir
# unit tests care only of standard stuff and parent dir
sys.path.append(".") 
sys.path.append("..") 


# debug stuff
import pdb

##cell
# support code
import tsc_datadict as tsc_dd
import parm_dict
from sklearn.utils import shuffle
from tensorflow.contrib.layers import flatten
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np

##cell
DD = tsc_dd.DataDict([ 'train', 'valid', 'test'], 'pickle', 'traffic-signs-data')

#load hyperparms
pd = parm_dict.parm_dict # reduce typing

def ldhp(key):
    ret = pd[key]
    print("%s -> %s" % (key, ret))
    return ret

rate = ldhp('learning_rate')
mu = ldhp('mu')
sigma = ldhp('sigma')
EPOCHS = ldhp('EPOCHS')
BATCH_SIZE=ldhp('BATCH_SIZE')
GOOD_ENOUGH=ldhp('GOOD_ENOUGH')

##cell@
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

n_train = DD.get_vbl('train', 'X').shape[0]
n_validation = DD.get_vbl('valid', 'X').shape[0]
n_test = DD.get_vbl('test', 'X').shape[0]
_X = DD.get_vbl('valid', 'X')
image_shape = "(" + str(_X.shape[1]) + ", " + str(_X.shape[2]) + ',' + str(_X.shape[3]) + ")"
n_classes = DD.n_classes

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

##cell
DD.show_sample_signs()
DD.show_distributions()

##cell
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

#already done in tsc_datadict in preprocess_images() method

##cell
# define these variables used in the model to reduce typing
X_train = DD.get_vbl('train', 'X')
y_train = DD.get_vbl('train', 'y')
X_validation = DD.get_vbl('valid', 'X')
y_validation = DD.get_vbl('valid', 'y')
X_test = DD.get_vbl('test', 'X')
y_test = DD.get_vbl('test', 'y')

##cell
###+++++++++ setting of hyperparms done at start, not here as in nc

##cell
def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the
    # weights and biases for each layer
    
    # conv strides: (batch, height, width, depth)
    # 2DO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu,
                                              stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1],
                           padding='VALID') + conv1_b

    # 2DO: Activation.
    conv1 = tf.nn.relu(conv1)

    # 2DO: Pooling. Input = 28x28x6. Output = 14x14x6
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='VALID')

    # 2DO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu,
                                              stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1],
                           padding='VALID') + conv2_b

    # 2DO: Activation.
    conv2 = tf.nn.relu(conv2)

    # 2DO: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='VALID')

    
    # 2DO: Flatten. Input = 5x5x16. Output = 400.
    # https://www.tensorflow.org/api_docs/python/tf/contrib/layers/flatten
    flat = flatten(conv2)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    # https://github.com/aymericdamien/TensorFlow-Examples/blob/master/\
    # examples/3_NeuralNetworks/convolutional_network.py
    # https://www.tensorflow.org/api_docs/python/tf/layers/dense
    fc1 = tf.layers.dense(flat, 120)
    
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.layers.dense(fc1, 84)
    
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.layers.dense(fc2, DD.n_classes)
    
    return logits

##cell
## Features and Labels
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, DD.n_classes)

## training pipeline

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

##cell
## model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

##cell
## train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        if validation_accuracy > GOOD_ENOUGH:
            print("better than good enough")
            break
        
    saver.save(sess, './lenet')
    print("Model saved")
    
##cell
##cell
##cell
##cell
##cell
##cell
##cell
##cell
##cell
