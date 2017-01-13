import tensorflow as tf
import numpy as np

class convLayer(object):
    '''
    Generates a convolutional layer with a 4d input and 4d output volume
    shape = [nrow, ncol, chanels, nfilters]
    strides = [1, nrow, ncol, 1]
    '''

    def __init__(self, shape, strides, name, padding="SAME"):
        self.shape = shape
        self.strides = strides
        self.name = name
        self.padding = padding

        # set weight var
        winit = tf.truncated_normal(shape, stddev=0.1)
        self.weight = tf.Variable(winit, name="b_{}".format(name))
        # set bias var
        binit = tf.constant(0.1, shape=[shape[-1]])
        self.bias = tf.Variable(binit, name="b_{}".format(name))

    def layer(self, x_in):
        weighted = tf.nn.conv2d(x_in, self.weight, strides=self.strides,
                                padding=self.padding)
        weighted_bias = tf.add(weighted, self.bias)

        return weighted_bias

    def layer_relu(self, x_in):
        # layer activations with relu and dropout
        activation = tf.nn.relu(self.layer(x_in))
        return activation


class fullLayer(object):
    '''
    Generates a traditional fully conected layer with a 4d input and 4d output volume
    shape = [nrow, ncol, chanels, nfilters]
    strides = [1, nrow, ncol, 1]
    '''

    def __init__(self, shape, name):
        self.shape = shape
        self.name = name
        # set weight var
        winit = tf.truncated_normal(shape, stddev=0.1)
        self.weight = tf.Variable(winit, name="b_{}".format(name))
        # set bias var
        binit = tf.constant(0.1, shape=[shape[-1]])
        self.bias = tf.Variable(binit, name="b_{}".format(name))

    def layer(self, x_in, flat=True):
        if flat:
            x_in = tf.reshape(x_in, [-1, self.shape[0]])

        weighted = tf.matmul(x_in, self.weight)
        weighted_bias = tf.add(weighted, self.bias)

        return weighted_bias

    def layer_relu(self, x_in, keep_prob=1):
        # layer activations with relu and dropout
        activation = tf.nn.dropout(tf.sigmoid(self.layer(x_in)), keep_prob)
        return activation

    def layer_sigmoid(self, x_in, keep_prob=1):
        # layer activations with sigmoid and dropout
        activation = tf.nn.dropout(tf.sigmoid(self.layer(x_in)), keep_prob)
        return activation


class maxPool(object):
    def __init__(self, ksize, strides):
        self.ksize = ksize
        self.strides = strides

    def pool(self, x):
        pooled = tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides,
                                padding='SAME')
        self.outdim = pooled.get_shape().as_list()
        self.numout = np.prod(self.outdim[-3:])
        return pooled


def seq_batch_iter(X, Y, batch_size, num_epochs, shuffle=True):
    data_size = len(X)
    if data_size != len(Y):
        raise ValueError("Differing data sizes")
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
    # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            X = X[shuffle_indices]
            Y = Y[shuffle_indices]

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            yield X[start_index:end_index], Y[start_index:end_index]


def random_batch_iter(X, Y, batch_size, num_epochs):
    if len(X) != len(Y):
        raise ValueError("Differing data sizes")

    for epoch in range(num_epochs):
        idx = np.random.choice(np.arange(len(X)), batch_size, replace=False)

        yield (X[idx], Y[idx])
