from __future__ import division

import random
import time

import numpy as np

from pos_tagging import mle, sample, viterbi, viterbi2, perceptron, phi_hmm, phi_alt
from helper_funcs import loss, get_data, frange


def test_mle():
    """
    Test the MLE function by measuring the log-likelihood
    of the test data given the trained model.
    """

    print '### TEST MLE ###'

    _, X, _, Y, suppx, suppy = get_data(k=1, n=1, shuffle=True)

    N = len(X)
    for percentage in frange(0.1, 1, 0.05):
        train_count = int(N * percentage)
        test_count = int((N-train_count) * 0.1)
        x_train = X[:train_count]
        y_train = Y[:train_count]
        x_test = X[-test_count:]
        y_test = Y[-test_count:]

        t, e = mle(x_train, y_train, suppx, suppy)
        ll = log_likelihood(x_test, y_test, t, e, suppx, suppy)
        print 'train percentage: %d%%, log-likelihood: %.5f' % (percentage * 100, ll)


def log_likelihood(X, Y, t, e, suppx, suppy):
    """
    Calculates and returns the log-likelihood of data, given an HMM model.
    :param X: an iterable over sequences of POS tags
    :param Y: a matching iterable over sequences of words
    :param t: the transition distributions of the model
    :param e: the emission distributions of the model
    :param suppx: the possible values for x variables.
    :param suppy: the possible values for y variables.
    """

    K = len(suppx)
    N = len(X)

    pos_idx = {v: k for k, v in enumerate(suppx)}
    words_idx = {v: k for k, v in enumerate(suppy)}

    ll = 0.0
    for x, y in zip(X, Y):
        for i in range(len(x)):
            x_prev = pos_idx[x[i-1]] if i > 0 else None
            x_curr = pos_idx[x[i]]
            y_curr = words_idx[y[i]]

            # Transmission
            if x_prev:
                ll += np.log(t[x_prev, x_curr])
            else:
                ll += np.log(1/K)

            # Emission
            ll += np.log(e[x_curr, y_curr])

    return ll / N


def test_sample(num_sentences):
    """
    Tests the sample function by measuring the log-likelihood
    of sampled data given an HMM model.
    :param num_sentences: The number of sentences to use for the test.
    """

    print '### TEST SAMPLE ###'

    MIN_LENGTH = 5
    MAX_LENGTH = 20

    _, x_train, _, y_train, suppx, suppy = get_data(k=1, n=1)
    t, e = mle(x_train, y_train, suppx, suppy)

    Ns = np.random.randint(low=MIN_LENGTH, high=MAX_LENGTH, size=num_sentences)
    X_gen, Y_gen = sample(Ns, suppx, suppy, t, e)

    ll = log_likelihood(X_gen, Y_gen, t, e, suppx, suppy)
    print 'log-likelihood of %d sampled instances: %.5f' % (num_sentences, ll)


def test_viterbi(num_sentences):
    """
    Tests the viterbi algorithm by sampling data, feeding it into the
    viterbi algorithm, and measuring the loss.
    :param num_sentences: The number of sentences to use for the test.
    """

    print '### TEST VITERBI ###'

    MIN_LENGTH = 5
    MAX_LENGTH = 20

    x_train, x_test, y_train, y_test, suppx, suppy = get_data()
    t, e = mle(x_train, y_train, suppx, suppy)

    Ns = np.random.randint(low=MIN_LENGTH, high=MAX_LENGTH, size=num_sentences)
    X_gen, Y_gen = sample(Ns, suppx, suppy, t, e)

    losses = []
    for x, y in zip(X_gen, Y_gen):
        x_hat = viterbi(y, suppx, suppy, t, e)
        losses.append(loss(x, x_hat))

    print 'viterbi loss with %d generated sentences: %f' % (num_sentences, sum(losses) / len(losses))


def test_perceptron(num_sentences):
    """
    Tests the Perceptron algorithm by running it with different
    parameters and comparing the outputs.
    :param num_sentences: The number of sentences to use for the test.
    """

    print '### TEST PERCEPTRON ###'

    LEARN_RATE = 0.35

    x_train, x_test, y_train, y_test, suppx, suppy = get_data(max_sentences=num_sentences)
    t, e = mle(x_train, y_train, suppx, suppy)

    phi, w = phi_hmm(suppx, suppy, e, t)

    params = [
        (np.zeros(w.shape), LEARN_RATE),
        (np.ones(w.shape), LEARN_RATE),
        (np.zeros(w.shape) + 1e-12, LEARN_RATE),
    ]

    W_perc = [w]

    for idx, (w0, rate) in enumerate(params):
        print 'learning w: %d/%d' % (idx + 1, len(params))
        W_epochs = perceptron(x_train, y_train, suppx, suppy, phi, w0, rate)
        W_perc.extend(W_epochs)

    for idx, w_perc in enumerate(W_perc):
        print 'testing w %d/%d' % (idx + 1, len(W_perc))
        losses = []
        for x, y in zip(x_test, y_test):
            x_hat = viterbi2(y, suppx, suppy, phi, w_perc)
            losses.append(loss(x, x_hat))

        print 'loss: %.5f' % np.mean(losses)


def compare_viterbi_viterbi2(num_sentences):
    """
    Compares the performance of the viterbi (HMM)
    and viterbi2 (log-linear) algorithms.
    :param num_sentences: The number of sentences to use for the test.
    """

    print '### COMPARE VITERBI WITH VITERBI2 ###'

    x_train, x_test, y_train, y_test, suppx, suppy = get_data(max_sentences=num_sentences)
    t, e = mle(x_train, y_train, suppx, suppy)
    phi, w = phi_hmm(suppx, suppy, e, t)

    viterbi_loss = []
    viterbi2_loss = []
    for x, y in zip(x_test, y_test):
        x_hat1 = viterbi(y, suppx, suppy, t, e)
        x_hat2 = viterbi2(y, suppx, suppy, phi, w)
        viterbi_loss.append(loss(x, x_hat1))
        viterbi2_loss.append(loss(x, x_hat2))

    print 'viterbi loss: %.5f' % np.mean(viterbi_loss)
    print 'viterbi2 loss: %.5f' % np.mean(viterbi2_loss)


def test_models(num_sentences):
    """
    Compares the performance of the HMM model and our alternative model.
    :param num_sentences: The number of sentences to use for the test.
    """

    print '### TEST MODELS ###'

    PERC_LEARN_RATE = 0.35
    PERC_EPOCHS = 1

    x_train, x_test, y_train, y_test, suppx, suppy = get_data(max_sentences=num_sentences)
    t, e = mle(x_train, y_train, suppx, suppy)

    models = [
        ('HMM', phi_hmm(suppx, suppy, e, t)),
        ('HMM Perceptron', phi_hmm(suppx, suppy, e, t)),
        ('Alternative', phi_alt(suppx, suppy))
    ]

    for model_name, (phi, w0) in models:
        print 'model: %s' % model_name,

        if model_name == 'HMM':
            w = w0
        elif model_name == 'HMM Perceptron':
            w = perceptron(x_train, y_train, suppx, suppy, phi, np.zeros(w0.shape) + 1e-12, PERC_LEARN_RATE, PERC_EPOCHS)[-1]
        else:
            w = perceptron(x_train, y_train, suppx, suppy, phi, w0, PERC_LEARN_RATE, PERC_EPOCHS)[-1]

        losses = []
        for x, y in zip(x_test, y_test):
            x_hat = viterbi2(y, suppx, suppy, phi, w)
            losses.append(loss(x, x_hat))

        print 'loss: %.5f' % np.mean(losses)


def test_features(num_sentences):
    """
    Compares the importance of each feature for the purpose of POS tagging.
    :param num_sentences: The number of sentences to use for the test.
    """

    print '### TEST FEATURES ###'

    PERC_LEARN_RATE = 0.35
    PERC_EPOCHS = 1

    x_train, x_test, y_train, y_test, suppx, suppy = get_data(max_sentences=num_sentences)

    features_names = [
        "Is digit?",
        "Is uppercase?",
        "Is first word?",
        "Is last word?",
        "Suffix of one letters",
        "Suffix of two letters",
        "Suffix of three letters",
        "After special words"
    ]

    models = [phi_alt(suppx, suppy, i+2) for i in xrange(len(features_names))]

    for model_name, (phi, w0) in zip(features_names, models):
        print 'feature: %s' % model_name,
        w = perceptron(x_train, y_train, suppx, suppy, phi, w0, PERC_LEARN_RATE, PERC_EPOCHS)[-1]

        losses = []
        for x, y in zip(x_test, y_test):
            x_hat = viterbi2(y, suppx, suppy, phi, w)
            losses.append(loss(x, x_hat))

        print 'loss: %.5f' % np.mean(losses)


def main():
    start_time = time.time()

    test_mle()
    test_sample(num_sentences=1000)
    test_viterbi(num_sentences=1000)
    test_perceptron(num_sentences=1000)
    compare_viterbi_viterbi2(num_sentences=1000)
    test_models(num_sentences=1000)
    test_features(num_sentences=2000)

    total_duration = time.time() - start_time
    print 'total run duration: %.5f seconds' % total_duration


if __name__ == '__main__':
    main()
