from matplotlib import pyplot
import tensorflow as tf
from data_collection import *
from network_elements import *
from psycopg2 import connect

# ---------------------------------------------------
# Network definition --------------------------------

def network(x, params):
    x = tf.expand_dims(x, 3)

    with tf.name_scope("convlayer"):
        # convolutional layer ----------------------
        # initialize convolutional layer
        convlayer = convLayer(shape=params['conv']['shape'],
                              strides=params['conv']['strides'], name="conv")
        h_conv = convlayer.layer_relu(x)

    with tf.name_scope("poollayer"):
        # pooling layer ----------------------------
        poollayer = maxPool(ksize=params['pool']['ksize'], strides=params['pool']['strides'])
        h_pool = poollayer.pool(h_conv)

    with tf.name_scope("fullayer"):
        # fully connected layer---------------------
        # initialize hidden fully connected layer
        fullayer = fullLayer(shape=[poollayer.numout, params['full']], name="full")
        h_full = fullayer.layer_relu(h_pool)

    with tf.name_scope("outlayer"):
        # output layer -----------------------------
        # initialize output layer
        outlayer = fullLayer(shape=params["vout"], name="out")
        v_out = outlayer.layer(h_full)
        # ------------------------------------------
    return v_out


# ---------------------------------------------------
# prepare cost and set optimization ops -------------

def optimization_ops(y, Y):
    '''Generates optimization related operations'''
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, Y))

    with tf.name_scope('optimization'):
        optimize = tf.train.AdamOptimizer(0.001)
        grads_vars = list(zip(tf.gradients(cost, tf.trainable_variables()), tf.trainable_variables()))
        train_step = optimize.apply_gradients(grads_and_vars=grads_vars)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # return cist
    return(cost, grads_vars, train_step, accuracy)


if __name__ == '__main__':

    # ---------------------------------------------------
    # project params ------------------------------------

    DAYS_BACK = 64
    NUM_CLASSES = 2

    # ---------------------------------------------------
    # data related parameters ---------------------------

    # financial instruments
    # yahoo ticker format
    INST = ['LUPE.ST', 'ABB.ST', 'ERIC-B.ST']
    # select time series and normalization
    # use either 'first' or 'return' optionally together with 'zscore'
    SERIES = {
    'close' :['first'],
    'open'  :['first'],
    'high'  :['first'],
    'low'   :['first'],
    'volume':['zscore']
    }

    # ---------------------------------------------------
    # network parameters --------------------------------

    NUM_HIDDEN_UNITS = 100
    NUM_FILTERS = 32
    POOL_WIDTH = 2

    NETWORK_PARAMS = {
        "conv" : {"shape" : [5,5,1,32], "strides" : [1,1,1,1]},
        "pool" : {"ksize" : [1,5,2,1], "strides" : [1,5,2,1]},
        "full" : NUM_HIDDEN_UNITS,
        "vout" : [NUM_HIDDEN_UNITS, NUM_CLASSES]
    }

    # generate datasets
    data = dataConstructor(series = SERIES, instruments = INST, days_forward=1, days_back=DAYS_BACK, conn = conn)
    data.generate_datasets(test_split="2015-06-01", flat = False)


    # input train vars
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, 5, DAYS_BACK], name="X")
        Y = tf.placeholder(tf.float32, [None, 2], name="Y")

        keep_prob = tf.placeholder(tf.float32)


    # ---------------------------------------------------
    # build network -------------------------------------


    with tf.name_scope('network'):
        y = network(X, NETWORK_PARAMS)

    with tf.name_scope('metrics'):
        cost, grads_vars, train_step, accuracy = optimization_ops(y, Y)

    # ---------------------------------------------------
    # set tensorboard ops  ------------------------------

    init_vars = tf.initialize_all_variables()

    # cost summary
    tf.scalar_summary('cost', cost)
    # acc summary
    tf.scalar_summary('accuracy', accuracy)
    # weights and biases summary
    for var in tf.trainable_variables():
        tf.histogram_summary(var.name, var)
    # gradient summary
    for grad, var in grads_vars:
        tf.histogram_summary(var.name + '/gradient', grad)

    merged_summaries = tf.merge_all_summaries()

    # ---------------------------------------------------
    # run operations  -----------------------------------


    step = 1
    test_interval = 10
    acc_test = []
    acc_train = []

    with tf.Session() as sess:
        sess.run(init_vars)
        summary_writer = tf.train.SummaryWriter("./summaries", graph=tf.get_default_graph())

        for batch in seq_batch_iter(data.xtrain, data.ytrain, 50, 10):
            x_batch, y_batch = batch

            _, summary = sess.run([train_step, merged_summaries], feed_dict={X: x_batch, Y: y_batch, keep_prob: 0.5})
            summary_writer.add_summary(summary, step)

            if step % test_interval == 0:
                acc_test.append(sess.run(accuracy, feed_dict={X: data.xtest, Y: data.ytest, keep_prob: 1.0}))
                acc_train.append(sess.run(accuracy, feed_dict={X: data.xtrain, Y: data.ytrain, keep_prob: 1.0}))

            if step % (test_interval) == 0:
                print("step: {} train_acc: {}, test_acc: {}".format(step, acc_train[-1], acc_test[-1]))

            step += 1

    pyplot.plot(acc_train)
    pyplot.plot(acc_test)



