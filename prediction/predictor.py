"""Simple MLP predictor. """
import tensorflow as tf
from prediction.layers import mlp


class MlpPredictor(object):
    """ Class for the MLP model. """

    def __init__(self, units, keep_prob):
        """
        Constructor for the MLP model.

        Args:
             units (list): list of integers, number of units in each layer
             keep_prob (float): probability of keeping each neuron
                during training (for dropout)
        """
        self.units = units
        self.x = tf.placeholder(shape=(None, None), dtype=tf.float32, name='x')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.similarity, self.params = mlp(self.x, units, keep_prob=self.keep_prob)
        self.prediction = tf.nn.softmax(self.similarity)
