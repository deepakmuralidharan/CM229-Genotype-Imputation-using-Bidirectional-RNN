'''
Course Project for CM229: Machine Learning for Bio-informatics

A Bidirectional Reccurent Neural Network (LSTM) implementation example using TensorFlow library for
genotype imputation

Authors: Deepak Muralidharan, Manikandan Srinivasan
'''
import tensorflow as tf
from tensorflow.python.ops.constant_op import constant
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
import time
import sys
import math

# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 56 # number of rows in the genotype dataset
display_step = 10 # for display purposes

# Network Parameters
n_input = 1 # dimension of input data (for genotype data-- since value is binary n_input = 1)
n_steps = 49 # number of columns in the data matrix
n_hidden = 5 # hidden layer -- hyperparameter -- values can range between 1-10
n_classes = 1 # dimension of outout (for genotype data modelling - output is 1 or 0)
max_epochs = 1000 # maximum number of epochs we want the training to run for

# loading the data file
data = np.loadtxt('/Users/deepakmuralidharan/Documents/Bidirectional-LSTM/data/train_data.txt',delimiter=',')

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
      x = raw_data[i * batch_size: (i + 1) * batch_size, 0:num_steps] # giving the entire range as time steps
      y = raw_data[i * batch_size: (i + 1) * batch_size, 1:(num_steps + 1)]
      yield (x,y)

def BiRNN(_X, _istate_fw, _istate_bw, _weights, _biases, _batch_size, _seq_len):

    # BiRNN requires to supply sequence_length as [batch_size, int64]
    # Note: Tensorflow 0.6.0 requires BiRNN sequence_length parameter to be set
    # For a better implementation with latest version of tensorflow, check below
    _seq_len = tf.fill([_batch_size], constant(_seq_len, dtype=tf.int64))

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    # need to change the input
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
                                            initial_state_bw=_istate_bw,
                                            sequence_length=_seq_len)

    # Linear activation
    # Get inner loop last output
    output = [tf.matmul(o, _weights['out']) + _biases['out'] for o in outputs]
    return output

pred = BiRNN(x, istate_fw, istate_bw, weights, biases, batch_size, n_steps)

# Define loss function and optimizer
pred = tf.concat(1, pred)
_y  = tf.squeeze(y,[2])
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, _y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:

    sess.run(init)

    for epoch in xrange(max_epochs):

        total_loss = []
        total_steps = sum(1 for x in geno_iterator(data, batch_size, n_steps))
        verbose = 10

        print 'Epoch {}'.format(epoch)
        start = time.time()
        for step, (batch_xs, batch_ys) in enumerate(
          geno_iterator(data, batch_size, n_steps)):
          batch_xs = np.reshape(batch_xs,[batch_size, n_steps, n_input])
          batch_ys = np.reshape(batch_ys,[batch_size, n_steps, n_input])
          #print batch_xs.shape
          #print batch_ys.shape

          sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys,
                                         istate_fw: np.zeros((batch_size, 2*n_hidden)),
                                         istate_bw: np.zeros((batch_size, 2*n_hidden))})
          predicted,loss,ground_truth = sess.run([pred,cost,_y], feed_dict={x: batch_xs, y: batch_ys,
                                           istate_fw: np.zeros((batch_size, 2*n_hidden)),
                                           istate_bw: np.zeros((batch_size, 2*n_hidden))})


          total_loss.append(loss)
          if verbose and step % verbose == 0:
              sys.stdout.write('\r{} / {} : loss = {}'.format(
                  step, total_steps, np.mean(total_loss)))
              sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
        print (1/(1+np.exp(-predicted[0,0:11])))
        print (ground_truth[0,0:11])

        print 'Training loss: {}'.format(np.mean(total_loss))

    print "Optimization Finished!"
    # Calculate accuracy for 128 mnist test images
    #test_len = 128
    #test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    #test_label = mnist.test.labels[:test_len]
    #print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                                             #istate_fw: np.zeros((test_len, 2*n_hidden)),
                                                             #istate_bw: np.zeros((test_len, 2*n_hidden))})
