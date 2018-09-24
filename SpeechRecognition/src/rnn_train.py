import tensorflow as tf

# for taking MFCC and label input
import rnn_input_data
import sound_constants

# for displaying elapsed time
import calendar as cal
import time
import sys
import os

# Training Parameters
num_input = 10 # mfcc data input
training_data_size = 8 # determines number of files in training and testing module
testing_data_size = num_input - training_data_size

# Network Parameters
learning_rate = 0.0001 # for large training set, it can be set 0.001
num_hidden = 250 # number of hidden layers
num_classes = 28 # total alphabet classes (a-z) + extra symbols (', ' ')
epoch = 50 # number of iterations
batch_size = 2 # number of batches

#shutting down debug logs
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

####################################################################################
mfcc_coeffs, text_data = rnn_input_data.mfcc_and_text_encoding()

class DataGenerator:
    def __init__(self, data_size):
        self.ptr = 0
        self.epochs = 0
        self.data_size = data_size

    def next_batch(self):
        if self.ptr > self.data_size:
            self.epochs += 1
            self.ptr = 0

        self.ptr += batch_size
        return mfcc_coeffs[self.ptr-batch_size : self.ptr], text_data[self.ptr-batch_size : self.ptr]

def reset_graph():
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()


def struct_network():

    reset_graph()

    input_data = tf.placeholder(tf.float32, [batch_size, sound_constants.MAX_ROW_SIZE_IN_DATA, sound_constants.MAX_COLUMN_SIZE_IN_DATA], name="train_input")
    target = tf.placeholder(tf.float32, [batch_size, sound_constants.MAX_ROW_SIZE_IN_TXT, sound_constants.MAX_COLUMN_SIZE_IN_TXT], name="train_output")

    keep_prob = tf.placeholder_with_default(1.0, [])


    fwd_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True, forget_bias=1.0)

    # creating one backward cell
    bkwd_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True, forget_bias=1.0)

    # creating bidirectional RNN
    val, _, _ = tf.nn.static_bidirectional_rnn(fwd_cell, bkwd_cell, tf.unstack(input_data), dtype=tf.float32)

    # adding dropouts
    val = tf.nn.dropout(val, keep_prob)

    val = tf.transpose(val, [1, 0, 2])
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    weight = tf.Variable(tf.truncated_normal([num_hidden * 2, sound_constants.MAX_ROW_SIZE_IN_TXT]))
    bias = tf.Variable(tf.constant(0.1, shape=[sound_constants.MAX_ROW_SIZE_IN_TXT]))


    # mapping to 28 output classes
    logits = tf.matmul(last, weight) + bias
    prediction = tf.nn.softmax(logits)
    prediction = tf.reshape(prediction, shape = [batch_size, sound_constants.MAX_ROW_SIZE_IN_TXT, sound_constants.MAX_COLUMN_SIZE_IN_TXT])


    # getting accuracy in training and testing model
    correct = tf.equal(prediction, target)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    logits = tf.reshape(logits, shape=[batch_size, sound_constants.MAX_ROW_SIZE_IN_TXT, sound_constants.MAX_COLUMN_SIZE_IN_TXT])


    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # returning components as dictionary elements
    return {'input_data' : input_data,
            'target' : target,
            'dropout': keep_prob,
            'loss': loss,
            'ts': train_step,
            'preds': prediction,
            'accuracy': accuracy
            }


def train_network(graph):


    # Settings for running tensorflow on GPU
    # tf_gpu_config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = True)
    # tf_gpu_config.gpu_options.allow_growth = True
    # tf_gpu_config.gpu_options.allocator_type = 'BFC'
    # tf_gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.9


    # with tf.Session(config = tf_gpu_config) as sess:
    with tf.Session() as sess:

        train_instance = DataGenerator(training_data_size)
        test_instance = DataGenerator(testing_data_size)

        sess.run(tf.global_variables_initializer())
        step, accuracy = 0, 0
        tr_losses, te_losses = [], []
        current_epoch = 0
        print (" ")
        print ("***********************************************************************************************")
        print ('*  Starting network training now. Displaying training and testing accuracy for each epoch !!  *')
        print ("***********************************************************************************************")
        print (" ")
        while current_epoch < epoch:
            start_time = cal.timegm(time.gmtime())
            step += 1
            trb = train_instance.next_batch()

            feed = {g['input_data'] : trb[0], g['target'] : trb[1], g['dropout'] : 0.6}
            accuracy_, _ = sess.run([g['accuracy'], g['ts']], feed_dict=feed)
            accuracy += accuracy_

            if train_instance.epochs > current_epoch:
                current_epoch += 1
                tr_losses.append(accuracy / step)
                step, accuracy = 0, 0

                #eval test set
                te_epoch = test_instance.epochs
                while test_instance.epochs == te_epoch:
                    step += 1
                    trc = test_instance.next_batch()
                    feed = {g['input_data']: trc[0], g['target']: trc[1]}
                    accuracy_ = sess.run([g['accuracy']], feed_dict=feed)[0]
                    accuracy += accuracy_

                te_losses.append(accuracy / step)
                step, accuracy = 0,0
                elapsed_time = cal.timegm(time.gmtime()) - start_time
                sys.stdout.write("Accuracy after epoch " + str(current_epoch) + " - Training: " + str(round(tr_losses[-1]*100, 3)) + "% - Testing: " + str(round(te_losses[-1]*100, 3)) + "%. ")
                mins, secs = divmod(elapsed_time, 60)
                hrs, mins = divmod(mins, 60)
                print ("Time taken = %02d:%02d:%02d" % (hrs, mins, secs))

    return tr_losses, te_losses

g = struct_network()
tr_losses, te_losses = train_network(g)
