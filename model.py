import tensorflow as tf
import numpy as np

import data_loader as dl

class Model:
    def __init__(self, train=False, fromCheckpoint=None):
        self.IMG_SIZE = 40
        self.NUM_LABEL = 10

        self.setupModel()

        self.dataset = dl.Dataset()

        if (train):
            self.setupTraining()
            self.dataset.loadData(train='train/', test='test/', categories=['sad', 'dead', 'at', 'hash', 'conf', 'empty', 'dot', 'dollar', 'plus', 'dash'])

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        if fromCheckpoint:
            self.saver.restore(self.sess, fromCheckpoint)
        else:
            init = tf.global_variables_initializer()
            self.sess.run(init)


    def setupModel(self):

        # INPUT 100x100 grayscale
        self.X = tf.placeholder(tf.float32, [None, self.IMG_SIZE, self.IMG_SIZE, 1])

        # 3 labels [plus, minus, mult]
        self.Y_ = tf.placeholder(tf.float32, [None, self.NUM_LABEL])

        # dropout
        self.pkeep = tf.placeholder(tf.float32);

        ###
        ### WEIGHTS
        ###

        W0 = tf.Variable(tf.truncated_normal([6, 6, 1, 6] ,stddev=0.1))
        B0 = tf.Variable(tf.ones([6]) / 10)

        W1 = tf.Variable(tf.truncated_normal([5, 5, 6, 12] ,stddev=0.1))
        B1 = tf.Variable(tf.ones([12]) / 10)

        W2 = tf.Variable(tf.truncated_normal([4, 4, 12, 24] ,stddev=0.1))
        B2 = tf.Variable(tf.ones([24]) / 10)

        W3 = tf.Variable(tf.truncated_normal([(self.IMG_SIZE / 4) * (self.IMG_SIZE / 4) * 24, 200] ,stddev=0.1))
        B3 = tf.Variable(tf.ones([200]) / 10)

        W4 = tf.Variable(tf.truncated_normal([200, self.NUM_LABEL] ,stddev=0.1))
        B4 = tf.Variable(tf.ones([self.NUM_LABEL]) / 10)

        ###
        ### LAYERS
        ###

        # 100x100 input image

        Y0 = tf.nn.conv2d(self.X, W0, strides=[1,1,1,1], padding='SAME');
        Y0d = tf.nn.relu(Y0 + B0);

        # 100x100 layer

        Y1 = tf.nn.conv2d(Y0d, W1, strides=[1,2,2,1], padding='SAME');
        Y1d = tf.nn.relu(Y1 + B1);

        # 50x50 layer

        Y2 = tf.nn.conv2d(Y1d, W2, strides=[1,2,2,1], padding='SAME');
        Y2d = tf.nn.relu(Y2 + B2);

        # 25x25

        Y3 = tf.matmul(tf.reshape(Y2d, [-1, (self.IMG_SIZE / 4) * (self.IMG_SIZE / 4) * 24]), W3)
        Y3d = tf.nn.relu(Y3 + B3)
        Y3dd = tf.nn.dropout(Y3d, self.pkeep)

        # Fully connected + Dropout

        self.YLogits = tf.matmul(Y3dd, W4) + B4
        self.Y = tf.nn.softmax(self.YLogits)


    def setupTraining(self):

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.YLogits, labels=self.Y_)
        self.cross_entropy = tf.reduce_mean(self.cross_entropy) * 100

        correct = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        self.training_rate = tf.placeholder(tf.float32);
        self.tr_max = 0.005
        self.tr_min = 0.0001
        self.train_step = tf.train.AdamOptimizer(self.training_rate).minimize(self.cross_entropy)


    def run_training(self, iterations, checkpointFolder='checkpoints', ident=0):
        for i in range(iterations+1):
            self.run_training_step(i, checkpointFolder, ident)


    def run_training_step(self, i, checkpointFolder, ident):
        if i % 10 == 0:
            # evaluate
            acc, loss = self.sess.run([self.accuracy, self.cross_entropy], feed_dict={self.X: self.dataset.test_images, self.Y_: self.dataset.test_labels, self.pkeep: 1})

            print('Step: ' + str(i) + '\tAccuracy: ' + str(acc) + '\tLoss: ' + str(loss))
        else:
            print('Step: ' + str(i))

        batch_x, batch_y = self.dataset.getTrainBatch(200)

        # train
        tr = self.tr_min + (self.tr_max - self.tr_min) * (np.exp(-i / 2000))

        self.sess.run(self.train_step, feed_dict={self.X: batch_x, self.Y_: batch_y, self.training_rate: tr, self.pkeep: 0.75})

        # Save
        if (i % 50 == 0):
            path = checkpointFolder + '/model_' + str(ident) + '_['+ str(i) + '].ckpt'
            p = self.saver.save(self.sess, path)
            print('Checkpoint saved ['+p+']')


    def run_predict(self, images):
        self.dataset.loadUnknown(images)

        Y = self.sess.run([self.Y], feed_dict={self.X: self.dataset.unknown_images, self.pkeep: 1})

        return Y;
