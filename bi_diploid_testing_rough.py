'''
GENOTYPE IMPUTATION ON DIPLOID DATA (contd...)

(Cleaned Version of the Code) - PART 2: Testing

Course Project for CM229: Machine Learning for Bio-informatics

A Bidirectional Reccurent Neural Network (LSTM) implementation example using TensorFlow library for
genotype imputation

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
import pickle

# Parameters
learning_rate = 0.01
training_iters = 100000
batch_size = 100

# Network Parameters
n_input = 3
n_steps = 50
n_hidden = 10
n_classes = 3
n_training = 1000
n_valid = 0
n_test = 92

data = np.loadtxt('data/geno_loc_new_diploid.txt',delimiter=',')

test_split  = np.copy(data[n_training: n_training + n_test, 0:n_steps])

test_input = np.copy(test_split[:,0:n_steps])
test_label = np.copy(test_split[:,0:n_steps])

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

pred_arg = []
for i in xrange(0, len(pred)):
    pred_arg.append(tf.argmax(pred[i],1))

pred_arg = tf.concat(0,pred_arg)

saver = tf.train.Saver()

with tf.Session() as sess:

    saver.restore(sess, './weights/diploid.bi.weights')
    print "restored..."
    mismatches = []



    for pos in range(0,50):
        truth_label = []
        predicted_label = []
        print pos

        for i in range(0, n_test):

            row_test_input = np.copy(test_input[i,:])

            x_b = row_test_input.astype(int)
            x_b = np.eye(n_input)[x_b]
            x_b = x_b.astype(float)
            x_b = np.reshape(x_b,[1, n_steps, n_classes])

            y_pred = sess.run(pred_arg, feed_dict={x: x_b,
                                                istate_fw: np.zeros((1, 2*n_hidden)),
                                                istate_bw: np.zeros((1, 2*n_hidden))})
            y_pred = np.asarray(y_pred)

            truth_label.append(test_input[i,pos])
            predicted_label.append(y_pred[pos])

            truth_label1 = np.asarray(truth_label)
            predicted_label1 = np.asarray(predicted_label)

        if pos == 0:
            ground = np.reshape(truth_label1,[n_test,1])
            predicted = np.reshape(predicted_label1,[n_test,1])
        else:
            ground = np.hstack((ground, np.reshape(truth_label1,[n_test,1])))
            predicted = np.hstack((predicted, np.reshape(predicted_label1,[n_test,1])))

        mismatches.append(sum(truth_label1 != np.around(predicted_label1)))

    print ground.shape
    print predicted.shape
    pickle.dump(ground, open("ground.p", "wb"))
    pickle.dump(predicted, open("predicted.p", "wb"))
