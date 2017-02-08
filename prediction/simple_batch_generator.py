""" This is to perform batch generation for a simple feature-based MLP model.
TODO: implement streaming.
"""
from random import shuffle
import numpy as np

NUM_CLASSES = 2


class SimpleBatchGenerator(object):
    """ Class to generate batches for a simple feature-based MLP. """
    def __init__(self, features, labels, ids):
        assert len(features) == len(labels) == len(ids)
        self.features = features
        labels = [l if l != 2 else 0 for l in labels]
        self.labels = self._convert_labels(labels)
        self.ids = ids

    @staticmethod
    def _convert_labels(labels):
        tmp = []
        for i in xrange(len(labels)):
            tmp.append((np.arange(NUM_CLASSES) == labels[i]).astype(float))
        return tmp

    def randomize_data(self):
        """ Randomizes the dataset. """
        index_shuf = range(len(self.ids))
        shuffled_feat = []
        shuffled_labels = []
        shuffled_ids = []
        shuffle(index_shuf)
        for i in index_shuf:
            shuffled_feat.append(self.features[i])
            shuffled_ids.append(self.ids[i])
            shuffled_labels.append(self.labels[i])
        self.features = shuffled_feat
        self.ids = shuffled_ids
        self.labels = shuffled_labels

    def _next_simple_batch(self, step, batch_size=32):
        offset = (step * batch_size) % len(self.features)
        batch_x = self.features[offset:offset + batch_size]
        batch_labels = self.labels[offset:offset + batch_size]
        batch_ids = self.ids[offset:offset + batch_size]
        return np.asarray(batch_x), np.asarray(batch_labels), batch_ids

    def batch_by_offset(self, start, end):
        """ Gets data by offset. """
        batch_x = self.features[start:end]
        batch_labels = self.labels[start:end]
        batch_ids = self.ids[start:end]
        return np.asarray(batch_x), np.asarray(batch_labels), batch_ids

    def next_batch(self, step, batch_size=None):
        """ Returns next batch. """
        return self._next_simple_batch(step, batch_size)
