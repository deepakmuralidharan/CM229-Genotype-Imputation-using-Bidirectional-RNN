'''
GENOTYPE IMPUTATION ON HAPLOID DATA

(Cleaned Version of the Code) - PART 1: Training

Course Project for CM229: Machine Learning for Bio-informatics

A Bidirectional Reccurent Neural Network (LSTM) implementation example using TensorFlow library for
genotype imputation

Authors: Deepak Muralidharan, Manikandan Srinivasan

Last edited: 5/29/2016
'''
import tensorflow as tf
from tensorflow.python.ops.constant_op import constant
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import time
import sys
import math
from sklearn.metrics import f1_score
import random
import matplotlib.pyplot as plt

# Parameters
learning_rate = 0.01
batch_size = 100
display_step = 10

# Network Parameters
n_input = 1
n_steps = 50
n_hidden = 10
n_classes = 1
max_epochs = 200
n_training = 2000
n_valid = 184

# loading data
data = np.loadtxt('data/geno_loc_new.txt',delimiter=',')
train_data = np.copy(data[0:n_training, 0:n_steps])
valid_split = np.copy(data[n_training:n_training + n_valid, 0:n_steps])

valid_input = np.copy(valid_split[:,0:n_steps])
valid_label = np.copy(valid_split[:,0:n_steps])

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input]) # [batch size, number of steps, input dimension]
# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
istate_fw = tf.placeholder("float", [None, 2*n_hidden]) # [batch size, 2 * number of hidden units]
istate_bw = tf.placeholder("float", [None, 2*n_hidden]) # [batch size, 2 * number of hidden units]
y = tf.placeholder("float", [None, n_steps, n_classes]) # [batch size, number of steps, number of classes (same size as x)]

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'hidden': tf.Variable(tf.random_normal([n_input, 2*n_hidden])), # [input dimension, 2 * number of hidden units]
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes])) # [2 * number of hidden units, number of classes]
}
biases = {
    'hidden': tf.Variable(tf.random_normal([2*n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def geno_iterator(raw_data, batch_size, num_steps):
  """
  Assume that raw_data is a numpy matrix of rows -- number of individuals (2184)
  and columns -- number of SNPs.

  Here the number of SNPs = number of columns = number of time steps.

  """

  col_iter = (raw_data.shape[0]) // batch_size # number of loops we would be needing

  for i  in range(col_iter):
      x = np.copy(raw_data[i * batch_size: (i + 1) * batch_size, 0:num_steps]) # giving the entire range as time steps
      y = np.copy(raw_data[i * batch_size: (i + 1) * batch_size, 0:num_steps])
      yield (x,y)


def BiRNN(_X, _istate_fw, _istate_bw, _weights, _biases):

     # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, _X,
                                            initial_state_fw=_istate_fw,
                                            initial_state_bw=_istate_bw)

    # Linear activation
    # Get inner loop last output
    output = [tf.matmul(o, _weights['out']) + _biases['out'] for o in outputs]
    return output


pred = BiRNN(x, istate_fw, istate_bw, weights, biases)
pred = tf.concat(1, pred)
_y  = tf.squeeze(y,[2])

# Define loss function and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, _y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:

    sess.run(init)

    best_train_epoch = float("inf")
    training_loss_arr = []
    validation_loss_arr = []

    for epoch in xrange(max_epochs):

        total_loss = []
        total_steps = sum(1 for x in geno_iterator(train_data, batch_size, n_steps))
        verbose = 10

        print 'Epoch {}'.format(epoch)

        for step, (batch_xs, batch_ys) in enumerate(
          geno_iterator(train_data, batch_size, n_steps)):

          batch_xs = np.reshape(batch_xs,[batch_size, n_steps, n_input])
          batch_ys = np.reshape(batch_ys,[batch_size, n_steps, n_input])

          sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                         istate_fw: np.zeros((batch_size, 2*n_hidden)),
                                         istate_bw: np.zeros((batch_size, 2*n_hidden))})

          predicted, loss, ground_truth = sess.run([pred, cost, _y], feed_dict={x: batch_xs, y: batch_ys,
                                           istate_fw: np.zeros((batch_size, 2*n_hidden)),
                                           istate_bw: np.zeros((batch_size, 2*n_hidden))})

          total_loss.append(loss)
          if verbose and step % verbose == 0:
              sys.stdout.write('\r{} / {} : loss = {}'.format(
                  step, total_steps, np.mean(total_loss)))
              sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')

        training_loss = np.mean(total_loss)
        print 'Training loss: {}'.format(training_loss)
        training_loss_arr.append(round(training_loss,2))

        valid_input = np.reshape(valid_input,[n_valid, n_steps, n_input])
        valid_label = np.reshape(valid_label,[n_valid, n_steps, n_input])

        validation_loss = sess.run(cost, feed_dict={x: valid_input, y: valid_label,
                                                                 istate_fw: np.zeros((n_valid, 2*n_hidden)),
                                                                 istate_bw: np.zeros((n_valid, 2*n_hidden))})

        print 'Validation loss: {}'.format(validation_loss)

        if training_loss < best_train_epoch and epoch > 90:
            saver.save(sess, './weights/haploid.bi.weights')
            best_train_epoch = training_loss


    plt.plot(range(1,201),np.asarray(training_loss_arr),label = 'Training Loss')
    plt.title('Training Loss vs Epochs [Bidirectional RNN on Haploid Data]')
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.legend(loc = 'best')
    plt.savefig('./results/bi_rnn_haploid_loss.png', bbox_inches='tight')

    plt.show()

    print "Optimization Finished!"
