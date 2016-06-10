'''
GENOTYPE IMPUTATION ON HAPLOID DATA (Part 2: Testing)

A Reccurent Neural Network (LSTM) implementation example
using TensorFlow library for genotype imputation.

(Cleaned Version of the Code) - PART 2: Testing
Course Project for CM229: Machine Learning for Bio-informatics
Authors: Deepak Muralidharan, Manikandan Srinivasan
Last edited: 5/28/2016
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
training_iters = 100000
batch_size = 100
display_step = 10

# Network Parameters
n_input = 1
n_steps = 49
n_hidden = 10
n_classes = 1

n_training = 2000
n_test = 184

data = np.loadtxt('data/geno_loc_new.txt',delimiter=',')
train_data = np.copy(data[0:n_training, 0:n_steps+1])
test_split  = np.copy(data[n_training: n_training + n_test, 0:n_steps+1])

test_input = np.copy(test_split[:,0:n_steps])
test_label = np.copy(test_split[:,1:n_steps+1])


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input]) # [batch size, number of steps, input dimension]
# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
istate = tf.placeholder("float", [None, 2*n_hidden]) # [batch size, 2 * number of hidden units]# [batch size, 2 * number of hidden units]
y = tf.placeholder("float", [None, n_steps, n_classes]) # [batch size, number of steps, number of classes (same size as x)]

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of foward + backward cells
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # [input dimension, 2 * number of hidden units]
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes])) # [2 * number of hidden units, number of classes]
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
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
      y = np.copy(raw_data[i * batch_size: (i + 1) * batch_size, 1:(num_steps + 1)])
      yield (x,y)


def RNN(_X, _istate, _weights, _biases):

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation
    _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, _X, initial_state=_istate)

    # Linear activation
    # Get inner loop last output
    output = [tf.matmul(o, _weights['out']) + _biases['out'] for o in outputs]
    return output


pred = RNN(x, istate, weights, biases)
pred = tf.concat(1, pred)
_y  = tf.squeeze(y,[2])

# Define loss function and optimizer
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, _y)) # Softmax loss

# Initializing the variables
saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, './weights/haploid.uni.weights')
    print "restored..."
    mismatches = []

    for pos in range(1,49):
        truth_label = []
        predicted_label = []
        print pos

        for i in range(0, n_test):

            row_test_input = np.copy(test_input[i,:])
            row_test_input[pos]=0

            row_test_input = np.reshape(row_test_input,[1, n_steps, n_input])
            y_pred = sess.run([pred], feed_dict={x: row_test_input,
                                                istate: np.zeros((1, 2*n_hidden))})
            y_pred = np.asarray(y_pred)
            y_pred = 1/(1+ np.exp(-y_pred))

            truth_label.append(test_input[i,pos])
            predicted_label.append(y_pred[0,0,pos-1])

        truth_label1 = np.asarray(truth_label)
        predicted_label1 = np.asarray(predicted_label)

        #print(truth_label)
        #print(predicted_label)

        mismatches.append(sum(truth_label1 != np.around(predicted_label1)))

    plt.stem(range(1,49),np.asarray(mismatches))
    axes = plt.gca()
    axes.set_ylim([0,100])
    plt.title('SNP position vs Mismatches (Haploid Data) [Unidirectional RNN]')
    plt.xlabel('SNP position')
    plt.ylabel('Number of Mismatches (out of 184)')
    plt.savefig('./results/uni_rnn_haploid.png', bbox_inches='tight')

    plt.show()
