""" Multilayer Perceptron model. """
import math
import tensorflow as tf


def _get_weights(dim_1, dim_2, scope, use_xavier=False):
    with tf.variable_scope(scope):
        if use_xavier:
            weights = tf.get_variable('weights', shape=[dim_1, dim_2],
                                      initializer=xavier_init(dim_1, dim_2))
            biases = tf.Variable(tf.ones([dim_2]), name='biases')
        else:
            weights = tf.Variable(tf.truncated_normal([dim_1, dim_2], stddev=1.0 / math.sqrt(
                float(dim_1))), name='weights')
            biases = tf.Variable(tf.zeros([dim_2]), name='biases')
    return weights, biases


def mlp(input_seq_tens, dims, keep_prob=0.9, use_xavier=False):
    """ Multilayer perceptron.

    Args:
        input_seq_tens (tf.Tensor): input tensor.
        dims (list): list of integers, number of units in each layer.
        keep_prob (float): dropout keep probability, default: 0.9.
        use_xavier (bool): if xavier initializer is used, otherwise
            weights are sampled from truncated normal distribution (default: False)
    """
    layer = 1
    all_weights = []
    assert len(dims) >= 2, 'dims should contain at least two values'
    for i in xrange(len(dims) - 2):
        weights, biases = _get_weights(dims[i], dims[i + 1], 'hidden_' + str(layer), use_xavier)
        all_weights.append(weights)
        all_weights.append(biases)
        input_seq_tens = tf.nn.relu((tf.matmul(input_seq_tens, weights) + biases),
                                    name='output_' + str(layer))
        input_seq_tens = tf.nn.dropout(input_seq_tens, keep_prob=keep_prob)
        layer += 1
    weights_last, biases_last = _get_weights(dims[-2], dims[-1], 'hidden_' + str(layer), use_xavier)
    output = tf.matmul(input_seq_tens, weights_last) + biases_last
    return output, all_weights


def xavier_init(num_inputs, num_outputs):
    """ Xavier initializer. """
    init_range = math.sqrt(6.0 / float(num_inputs + num_outputs))
    return tf.random_uniform_initializer(-init_range, init_range)
