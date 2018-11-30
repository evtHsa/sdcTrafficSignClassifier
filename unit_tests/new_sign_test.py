#!/usr/bin/env python3
#

import sys

# unit tests are usually run from their own dir or parent dir
# unit tests care only of standard stuff and parent dir
sys.path.append(".") 
sys.path.append("..") 

rate = 0.001 
mu = 0 
sigma = 0.1
EPOCHS = 30
BATCH_SIZE = 2
GOOD_ENOUGH = 0.935

# debug stuff
import pdb

# support code
import numpy as np
import tsc_datadict as tsc_dd
import tensorflow as tf

# https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/

from tensorflow.contrib.layers import flatten

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

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    ret = total_accuracy / num_examples
    return ret


## Features and Labels
DD = tsc_dd.DataDict([ 'test'], 'image_dir', 'found_signs')
DD.summarize()
DD.show_sample_signs()
DD.show_distributions()
X_test = DD.get_vbl('test', 'X')
y_test = DD.get_vbl('test', 'y')
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, DD.n_classes)

logits = LeNet(x)

prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver=tf.train.Saver()
## Predict Sign Types
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    predictions = sess.run(prediction, feed_dict={x: X_test})
    
print('\nPrediction   Ground Truth')
print('----------   ------------')    
for p_i, y_i in zip(predictions, y_test):
    print('    {0:2d}            {1:2d}     '.format(p_i, y_i))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    acc = evaluate(X_test, y_test)
    print("accuracy = %.2ff" % acc)

## Calculate the accuracy for the download images as a percent correct
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    new_accuracy = evaluate(X_test, y_test)
    print("\nNew Accuracy = {:.3f}".format(new_accuracy))

#output top 5 softmax probabilities
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    softmaxes = sess.run(tf.nn.softmax(logits), feed_dict={x: X_test})
    top5, _ = sess.run(tf.nn.top_k(softmaxes, k=5))
    
np.set_printoptions(precision=5)
print('\n' + str(top5))

print("done")
