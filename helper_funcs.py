from __future__ import division

import sys
from numpy import random

import numpy as np

import parse


def loss(x, x_hat):
    """
    Calculates and returns the loss of two given POS tags.
    :param x: The correct POS tag.
    :param x_hat: The inferred POS tag.
    """

    T = len(x)
    return sum([1 for i in xrange(T) if x[i] != x_hat[i]]) / T


def avg_loss(X, X_hat):
    """
    Calculates and returns the average loss two given POS tag sets.
    :param X: The correct POS tags set.
    :param X_hat: The inferred POS tags set.
    """

    return np.mean([loss(x, x_hat) for x, x_hat in zip(X, X_hat)])


def dot(v1, v2):
    """
    Calculates and returns the dot product of a sparse vector
    containing only 0/1 values, and a full vector.
    :param v1: The sparse vector (contains indexes of '1's)
    :param v2: The full vector.
    """

    return sum(v2[i] for i in v1)


def frange(x, y, jump):
    """
    Generator for range of floating numbers.
    :param x: The start value of the generator.
    :param y: The end value of the generator.
    :param jump: The increment size of each generated value.
    """

    while x < y:
        yield x
        x += jump


def get_data(k=5, n=1, shuffle=True, max_sentences=sys.maxint):
    """
    Reads and returns the input data along with the
    sets of unique words and POS tags.
    :param shuffle: Should the data be shuffled?
    :param k: cross-validation factor, if set to e.g. 4, size of test set is 1/4 of data
    :param n: number of train/test sets to draw
    :param max_sentences: maximal number of sentences to read from file
    :return:
     x_train - Train set of POS tags.
     x_test - Test set of POS tags.
     y_train - Train set of sentences.
     y_test - Test set of sentences.
     suppx - a set of the values of X (parts of speech)
     suppy - a set of the values of Y (words)
    """

    data, xvals, yvals = parse.collect_sets("data_split.gz", k=k, n=n, max_sentences=max_sentences)

    if shuffle:
        random.shuffle(data['train'])
        random.shuffle(data['test'])

    train_data = data['train']
    x_train = [s[0] for s in train_data]
    y_train = [s[1] for s in train_data]

    test_data = data['test']
    x_test = [s[0] for s in test_data]
    y_test = [s[1] for s in test_data]

    suppx = sorted(list(xvals))
    suppy = sorted(list(yvals))

    return x_train, x_test, y_train, y_test, suppx, suppy
