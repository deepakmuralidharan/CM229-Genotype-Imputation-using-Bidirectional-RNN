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
from sklearn.metrics import f1_score
import random
import matplotlib.pyplot as plt


# Parameters
learning_rate = 0.01
training_iters = 100000
batch_size = 56 # number of rows in the genotype dataset
display_step = 10 # for display purposes

# Network Parameters
n_input = 1 # dimension of input data (for genotype data-- since value is binary n_input = 1)
n_steps = 50 # number of columns in the data matrix
n_hidden = 10 # hidden layer -- hyperparameter -- values can range between 1-10
n_classes = 1 # dimension of outout (for genotype data modelling - output is 1 or 0)
# loading the data file
n_training = 2016
n_valid = 48
#n_test = 2084 - n_valid - n_training
n_test = 100

data = np.loadtxt('data/geno_good_data.txt',delimiter=',')

test_split  = np.copy(data[n_training + n_valid: n_training + n_valid + n_test, 0:n_steps+1])
#test_split = np.copy(data[0:100, 0:n_steps+1])


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


#pred = BiRNN(x, istate_fw, istate_bw, weights, biases, batch_size, n_steps)
pred = BiRNN(x, istate_fw, istate_bw, weights, biases)
pred = tf.concat(1, pred)

# Define loss function and optimizer

_y  = tf.squeeze(y,[2])
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, _y)) # Softmax loss
#cost_valid = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, _y)) # for test


saver = tf.train.Saver()


with tf.Session() as sess:

    saver.restore(sess, './geno_bi.weights')
    print "restored..."
    mismatches = []
    for pos in range(2,49):
        truth_label = []
        predicted_label = []
        print pos
        for i in range(0, n_test):

            #print 'Impute data row number: {}'.format(i)


            row_test_input = np.copy(test_input[i,:])
            row_test_input[pos]=0

            row_test_input = np.reshape(row_test_input,[1, n_steps, n_input])
            y_pred = sess.run([pred], feed_dict={x: row_test_input,
                                                istate_fw: np.zeros((1, 2*n_hidden)),
                                                istate_bw: np.zeros((1, 2*n_hidden))})
            y_pred = np.asarray(y_pred)
            y_pred = 1/(1+ np.exp(-y_pred))
            #print y_pred[0,0,pos]
            #print test_input[i,pos]
            truth_label.append(test_input[i,pos])
            predicted_label.append(y_pred[0,0,pos])

        truth_label1 = np.asarray(truth_label)
        predicted_label1 = np.asarray(predicted_label)

        #print(truth_label)
        #print(predicted_label)
        #print type(mismatches)
        mismatches.append(sum(truth_label1 != np.around(predicted_label1)))

    plt.stem(range(2,49),np.asarray(mismatches))
    plt.show()
