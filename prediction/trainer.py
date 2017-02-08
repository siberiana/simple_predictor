"""Methods for training an MLP."""
from __future__ import print_function

import warnings
import tensorflow as tf

from prediction.predictor import MlpPredictor

MARGIN = 0.5
POS_DIM = 1
NUM_LABELS = 2


class Trainer(object):
    """
            Class for training the MLP model.

            Args:
            model (MlpPredictor): model to be trained
            optimizer (str): optimizer to be used (Adagrad, SGD, Momentum, Adam), default: SGD.
            loss (str): loss function (cross-entropy, log or hinge), default: log
            learning_rate (float): initial learning rate, default: 0.05
            reg_rate (float): L2 regularization rate, default: 1e-5
            momentum (float): momentum, needed only if momentum optimizer is used, default: 0.9.
    """
    def __init__(self, model, optimizer='SGD', loss='log', learning_rate=0.005, reg_rate=1e-5,
                 momentum=0.9):
        self.model = model
        assert isinstance(model, MlpPredictor)
        self.labels = tf.placeholder(tf.float32, shape=(None, NUM_LABELS), name='labels')

        # adding L2 regularization
        params = tf.trainable_variables()
        l2_loss = sum([tf.nn.l2_loss(var) for var in params]) * reg_rate
        self.loss = reg_rate * l2_loss

        # setting the loss
        if loss.lower() == 'cross-entropy':
            self.loss += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                self.model.similarity, self.labels))
        elif loss.lower() == 'log':
            self.loss += tf.contrib.losses.log_loss(self.model.prediction, self.labels)
        elif loss.lower() == 'hinge':
            self.loss += tf.contrib.losses.hinge_loss(self.model.prediction, self.labels)
        elif loss.lower() == 'pairwise':
            # TODO: check this implementation
            positives = tf.mul(self.labels, self.model.prediction)
            reversed_labels = tf.add(tf.ones_like(self.labels), tf.scalar_mul(-1, self.labels))
            negatives = tf.mul(reversed_labels, self.model.prediction)
            _, positives = tf.split(POS_DIM, NUM_LABELS, positives)
            _, negatives = tf.split(POS_DIM, NUM_LABELS, negatives)
            mm_loss = tf.maximum(0., MARGIN - positives + negatives)
            self.loss += tf.reduce_mean(mm_loss)
        else:
            warnings.warn('Unknown loss function, using cross-entropy')
            self.loss += tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                self.model.similarity, self.labels))

        # setting the optimizer
        if optimizer.lower() == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate).minimize(self.loss)
        elif optimizer.lower() == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(
                self.loss)
        elif optimizer.lower() == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(
                self.loss)
        elif optimizer.lower() == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                        momentum=momentum).minimize(self.loss)
        elif optimizer.lower() == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        else:
            warnings.warn('Unknown optimizer, using SGD instead')
            self.optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate).minimize(self.loss)

    def _step(self, session, batch_x, batch_labels, train):
        feed_dict = {self.model.x: batch_x,
                     self.labels: batch_labels}
        if train:
            _, loss, predictions = session.run([self.optimizer, self.loss,
                                                self.model.prediction], feed_dict=feed_dict)
        else:
            loss, predictions = session.run([self.loss, self.model.prediction],
                                            feed_dict=feed_dict)
        return loss, predictions

    def training_step(self, session, batch_x, batch_labels):
        """ training step """
        loss, predictions = self._step(session, batch_x, batch_labels, train=True)
        return loss, predictions

    def eval_step(self, session, batch_x, batch_labels):
        """ evaluation step (no gradient updates) """
        loss, predictions = self._step(session, batch_x, batch_labels, train=False)
        return loss, predictions
