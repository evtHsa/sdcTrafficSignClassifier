import tensorflow as tf

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
