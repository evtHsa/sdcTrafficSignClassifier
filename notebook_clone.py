#!/usr/bin/env python3
#

# debug stuff
import pdb

# support code
import tsc_util as tscu
import tsc_datadict as tsc_dd
from sklearn.utils import shuffle

#step 0
# Load pickled data
import pickle
    
# TODO: Fill this in based on where you saved the training and testing data
#DD = tsc_dd.DataDict(show_sample=True, show_distrib=True)
DD = tsc_dd.DataDict()

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np

# TODO: Number of training examples
n_train = DD.get_vbl('train', 'X').shape[0]

n_validation = DD.get_vbl('valid', 'X').shape[0]

# TODO: Number of testing examples.
n_test = DD.get_vbl('test', 'X').shape[0]

# 2DO: What's the shape of an traffic sign image?
_X = DD.get_vbl('valid', 'X')
image_shape = "(" + str(_X.shape[1]) + "," + str(_X.shape[2]) + ")"

# 2DO: How many unique classes/labels there are in the dataset.
# now part of dd

print("shape(features) = " + str(DD.get_vbl('train', 'X').shape ))
print("shape(labels) = " + str(DD.get_vbl('train', 'y').shape ))
print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", DD.n_classes)

from tensorflow.contrib.layers import flatten
import tensorflow as tf

X_train = DD.get_vbl('train', 'X')
y_train = DD.get_vbl('train', 'y')
X_validation = DD.get_vbl('valid', 'X')
y_validation = DD.get_vbl('valid', 'y')
X_test = DD.get_vbl('test', 'X')
y_test = DD.get_vbl('test', 'y')

## setup tensorflow
import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128

# ??: options for implementing
#    1) simple adaptation of linear script in notebook_clone.py
#            - easiest and fastest
#           - 
#
#    2) class approach with train & eval methods
#            - easier to adapt and parameterize for more sophisticated approaches
#            - scoping issues when we deviate from ipynb
#            - maybe later

## implement LeNet
from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the
    # weights and biases for each layer
    mu = 0
    sigma = 0.1
    
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
    logits = tf.layers.dense(fc2, 10)
    
    return logits

## Features and Labels
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

## training pipeline
rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

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
            pdb.set_trace()
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
    assert("FIXME: validate pasted code below here" == None)

## evaluate the model
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

