"""
*** FOR EXAMPLE PURPOSES ***
"""
'''
import tensorflow as tf
import numpy as np  # linear algebra
import preprocessor
import models
import config
import math

from tqdm import tqdm

tf.set_random_seed(0)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def run():

    data = preprocessor.read_data(config.train_csv_path)

    print('*** Importing images ***')

    x_train = []
    y_train = []

    for item in tqdm(data):
        x_train.append(models.AnnotatedRecord.medium_image(item))
        y_train.append(models.AnnotatedRecord.some_hot(item))

    x_data = np.array(x_train, np.float16) / 255.
    y_data = np.array(y_train, np.uint8)

    print(x_data.shape)
    print(y_data.shape)

    split = 35000
    x_train = x_data[:split]
    x_valid = x_data[split:]
    y_train = y_data[:split]
    y_valid = y_data[split:]

    print('hello world')

    X = tf.placeholder(tf.float32, [None, 64, 64, 3])

    Y_ = tf.placeholder(tf.float32, [None, 17])

    # variable learning rate
    lr = tf.placeholder(tf.float32)

    # three convolutional layers with their channel counts, and a
    # fully connected layer (tha last layer has 17 softmax neurons)
    K = 4  # first convolutional layer output depth
    L = 8  # second convolutional layer output depth
    M = 12  # third convolutional layer
    N = 200  # fully connected layer

    W1 = tf.Variable(tf.truncated_normal([5, 5, 3, K], stddev=0.1))  # 5x5 patch, 3 input channel, K output channels
    B1 = tf.Variable(tf.ones([K]) / 17)
    W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
    B2 = tf.Variable(tf.ones([L]) / 17)
    W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
    B3 = tf.Variable(tf.ones([M]) / 17)

    W4 = tf.Variable(tf.truncated_normal([16 * 16 * M, N], stddev=0.1))
    B4 = tf.Variable(tf.ones([N]) / 17)
    W5 = tf.Variable(tf.truncated_normal([N, 17], stddev=0.1))
    B5 = tf.Variable(tf.ones([17]) / 17)

    # The model
    stride = 1  # output is 64x64
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    stride = 2  # output is 32x32
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    stride = 2  # output is 16x16
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 16 * 16 * M])

    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
    Ylogits = tf.matmul(Y4, W5) + B5
    Y = tf.nn.softmax(Ylogits)

    # cross-entropy loss function (= -sum(Y_i * log(Yi)) ), normalised for batches of 100  images
    # TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
    # problems with log(0) which is NaN
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
    cross_entropy = tf.reduce_mean(cross_entropy)*100

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # training step, the learning rate is a placeholder
    train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    # init
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # You can call this function in a loop to train the model, 100 images at a time

    def training_step(i):

        # training on batches of 100 images with 100 labels
        batch_X = x_train[:100]
        batch_Y = y_train[:100]

        # learning rate decay
        max_learning_rate = 0.003
        min_learning_rate = 0.0001
        decay_speed = 2000.0
        learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)

        # the backpropagation training step
        sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate})

        print(sess.run(accuracy))

    for i in range(1):
        training_step(i)
'''