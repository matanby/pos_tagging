from __future__ import division

import sys

import numpy as np

import parse


def loss(x, x_hat):
    T = len(x)
    return sum([1 for i in xrange(T) if x[i] != x_hat[i]]) / T


def avg_loss(X, X_hat):
    return np.mean([loss(x, x_hat) for x, x_hat in zip(X, X_hat)])


def dot(phi, w):
    return sum(w[i] for i in phi)


def frange(x, y, jump):
    while x < y:
        yield x
        x += jump


def get_data(k=5, n=1, max_sentences=sys.maxint):
    data, xvals, yvals = parse.collect_sets("data_split.gz", k=k, n=n, max_sentences=max_sentences)

    train_data = data['train']
    x_train = [s[0] for s in train_data]
    y_train = [s[1] for s in train_data]

    test_data = data['test']
    x_test = [s[0] for s in test_data]
    y_test = [s[1] for s in test_data]

    suppx = sorted(list(xvals))
    suppy = sorted(list(yvals))

    return x_train, x_test, y_train, y_test, suppx, suppy
