from __future__ import division

import random

import numpy as np

from helper_funcs import dot

from itertools import product
from string import ascii_lowercase


def mle(X, Y, suppx, suppy):
    """
    Calculate the maximum likelihood estimators for the transition and
    emission distributions, in the multinomial HMM case.
    : param x : an iterable over sequences of POS tags
    : param y : a matching iterable over sequences of words
    : param suppx : the possible values for x variables.
    : param suppy : the possible values for y variables.
    : return : a tuple (t, e), with:
    t.shape = (|val(X)|, |val(X)|), and
    e.shape = (|val(X)|, |val(Y)|)
    """

    pos_idx = {v: k for k, v in enumerate(suppx)}
    words_idx = {v: k for k, v in enumerate(suppy)}

    t = np.zeros([len(suppx), len(suppx)]) + 1e-12
    e = np.zeros([len(suppx), len(suppy)]) + 1e-12

    for x in X:
        for k in xrange(1, len(x)):
            i = pos_idx[x[k - 1]]
            j = pos_idx[x[k]]
            t[i, j] += 1

    for x, y in zip(X, Y):
        for pos, word in zip(x, y):
            i = pos_idx[pos]
            j = words_idx[word]
            e[i, j] += 1

    t = t / t.sum(axis=1)[:, None]
    e = e / e.sum(axis=1)[:, None]

    # TODO: remove!
    # t = t.clip(min=1e-12)
    # e = e.clip(min=1e-12)

    return t, e


def sample(Ns, suppx, suppy, t, e):
    """
    sample sequences from the model.
    : param Ns : a vector with desired sample lengths, a sample is
    generated per entry in the vector , with corresponding length.
    : param suppx : the possible values for x variables, ordered as in t, and e
    : param suppy : the possible values for y variables, ordered as in e
    : param t : the transition distributions of the model
    : param e : the emission distributions of the model
    : return : x, y - two iterables describing the sampled sequences.
    """

    suppx_range = range(len(suppx))
    suppy_range = range(len(suppy))

    x = []
    y = []

    for N in Ns:
        x_i = []
        y_i = []

        for i in xrange(0, N):
            if i == 0:
                curr_pos_idx = random.choice(suppx_range)
            else:
                prev_pos_idx = x_i[i - 1]
                curr_pos_idx = np.random.choice(suppx_range, p=t[prev_pos_idx, :])

            curr_word_idx = np.random.choice(suppy_range, p=e[curr_pos_idx, :])

            x_i.append(curr_pos_idx)
            y_i.append(curr_word_idx)

        x.append([suppx[pos_idx] for pos_idx in x_i])
        y.append([suppy[word_idx] for word_idx in y_i])

    return x, y


def viterbi(y, suppx, suppy, t, e):
    """
    Calculate the maximum a-posteriori assignment of x's.
    : param y : a sequence of words
    : param suppx : the support of x (what values it can attain)
    : param suppy : the support of y (what values it can attain)
    : param t : the transition distributions of the model
    : param e : the emission distributions of the model
    : return : xhat, the most likely sequence of hidden states (parts of speech).
    """

    K = len(suppx)
    T = len(y)

    words_idx = {v: k for k, v in enumerate(suppy)}

    T1 = np.zeros([K, T])
    T2 = np.zeros([K, T])

    for i in xrange(K):
        T1[i, 0] = e[i, words_idx[y[0]]]
        T2[i, 0] = 0

    for i in xrange(1, T):
        for j in xrange(K):
            T1[j, i] = max(
                T1[k, i - 1] * t[k, j]
                for k in xrange(K)
            )
            T1[j, i] *= e[j, words_idx[y[i]]]
            T2[j, i] = np.argmax([
                 T1[k, i - 1] * t[k, j]
                 for k in xrange(K)
            ])

    xhat = []
    z_curr = int(np.argmax(T1[:, T - 1]))
    xhat.insert(0, suppx[z_curr])

    for i in xrange(T - 1, 0, -1):
        z_curr = int(T2[z_curr, i])
        xhat.insert(0, suppx[z_curr])

    return xhat


def viterbi2(y, suppx, suppy, phi, w):
    """
    Calculate the assignment of x that obtains the maximum log - linear score.
    : param y : a sequence of words.
    : param suppx : the support of x (what values it can attain).
    : param phi : a mapping from (x_t, x_{t+1} , y_{1..t+1} to indices of w.
    : param w : the linear model.
    : return : xhat, the most likely sequence of hidden states (parts of speech).
    """

    K = len(suppx)
    T = len(y)

    T1 = np.zeros([K, T])
    T2 = np.zeros([K, T])

    for i in xrange(K):
        phi_idxs = phi(None, i, y, 0)
        T1[i, 0] = float(np.exp(dot(phi_idxs, w)))
        T2[i, 0] = 0

    for i in xrange(1, T):
        for j in xrange(K):
            T2[j, i] = np.argmax([
                 T1[k, i - 1] * float(np.exp(dot(phi(k, j, y, i), w)))
                 for k in xrange(K)
             ])

            argmax = int(T2[j, i])
            T1[j, i] = T1[argmax, i - 1] * float(np.exp(dot(phi(argmax, j, y, i), w)))

    xhat = []
    z_curr = int(np.argmax(T1[:, T - 1]))
    xhat.insert(0, suppx[z_curr])

    for i in xrange(T - 1, 0, -1):
        z_curr = int(T2[z_curr, i])
        xhat.insert(0, suppx[z_curr])

    return xhat


def perceptron(X, Y, suppx, suppy, phi, w0, rate, epochs=1):
    """
    Find w that maximizes the log - linear score
    : param X : POS tags for sentences (iterable of lists of elements in suppx).
    : param Y : words in respective sentences (iterable of lists of words in suppy).
    : param suppx : the support of x (what values it can attain).
    : param suppy : the support of y (what values it can attain).
    : param phi : a mapping from (None | x_1 , x_2 , y_2 to indices of w.
    : param w0 : initial model.
    : param rate : rate of learning.
    : return : w, a weight vector for the log - linear model features.
    """

    N = len(X)
    pos_idx = {v: k for k, v in enumerate(suppx)}
    W_epochs = []

    for epoch in xrange(epochs):
        W = [W_epochs[-1]] if W_epochs else [w0]

        for i in xrange(N):
            # TODO: remove
            if i % 100 == 0:
                print 'i:', i

            X_i = X[i]
            Y_i = Y[i]
            ni = len(X_i)
            w = W[i].copy()
            # TODO: remove!
            # X_i_hat = viterbi2(Y_i, suppx, suppy, phi, w)
            X_i_hat = viterbi2(Y_i, suppx, suppy, phi, np.log(w))

            for t in xrange(ni):
                if t != 0:
                    s_prev = pos_idx[X_i[t - 1]]
                    s_hat_prev = pos_idx[X_i_hat[t - 1]]
                else:
                    s_prev = None
                    s_hat_prev = None

                s_curr = pos_idx[X_i[t]]
                s_hat_curr = pos_idx[X_i_hat[t]]

                phi_idxs = phi(s_prev, s_curr, Y_i, t)
                w[phi_idxs] += (1 * rate)

                phi_idxs = phi(s_hat_prev, s_hat_curr, Y_i, t)
                w[phi_idxs] -= (1 * rate)

            w = w.clip(min=1e-30)
            W.append(w)

        W_epochs.append(sum(W) / N)

    # TODO: remove!
    # return W_epochs
    return np.log(W_epochs)


def phi_hmm(suppx, suppy, e, t):
    K = len(suppx)
    T = len(suppy)
    words_idx = {v: k for k, v in enumerate(suppy)}

    def phi(x_prev, x_curr, Y, t):
        phi_idxs = []

        if x_prev is not None:
            trans_idx = x_prev * K + x_curr
            phi_idxs.append(trans_idx)

        emiss_idx = K ** 2 + x_curr * T + words_idx[Y[t]]
        phi_idxs.append(emiss_idx)
        return phi_idxs

    w = np.log(np.concatenate((t.reshape(K ** 2, 1), e.reshape(K * T, 1))) + 1e-12)
    return phi, w


def phi_alt(suppx, suppy, current_feature=None):
    K = len(suppx)
    T = len(suppy)
    words_idx = {v: k for k, v in enumerate(suppy)}

    keywords_1 = [''.join(i) for i in product(ascii_lowercase, repeat=1)]
    keywords_2 = [''.join(i) for i in product(ascii_lowercase, repeat=2)]
    keywords_3 = [''.join(i) for i in product(ascii_lowercase, repeat=3)]
    suffix_idx_1 = {v: k for k, v in enumerate(keywords_1)}
    suffix_idx_2 = {v: k for k, v in enumerate(keywords_2)}
    suffix_idx_3 = {v: k for k, v in enumerate(keywords_3)}
    special_prev_word_list = ["is", "a", "and", "the", '"', "it"]
    special_prev_word_dict = {v: k for k, v in enumerate(special_prev_word_list)}

    def suffix_1_leters(x_prev, x_curr, Y, t):
        if len(Y[t]) < 1:
            return -1
        elif Y[t][-1:] not in suffix_idx_1:
            return -1
        else:
            return (suffix_idx_1[Y[t][-1:]] + 1) * K + x_curr

    def suffix_2_leters(x_prev, x_curr, Y, t):
        if len(Y[t]) < 2:
            return -1
        elif Y[t][-2:] not in suffix_idx_2:
            return -1
        else:
            return (suffix_idx_2[Y[t][-2:]] + 1) * K + x_curr

    def suffix_3_leters(x_prev, x_curr, Y, t):
        if len(Y[t]) < 3:
            return -1
        elif Y[t][-3:] not in suffix_idx_3:
            return -1
        else:
            return (suffix_idx_3[Y[t][-3:]] + 1) * K + x_curr

    all_features = [
        # Transition
        (
            lambda x_prev, x_curr, Y, t: x_prev * K + x_curr if x_prev is not None else -1,
            K ** 2
        ),

        # Emission
        (
            lambda x_prev, x_curr, Y, t: x_curr * T + words_idx[Y[t]],
            K * T
        ),

        # Is digit?
        (
            lambda x_prev, x_curr, Y, t: x_curr if Y[t].isdigit() else -1,
            K
        ),
        # Is uppercase?
        (
            lambda x_prev, x_curr, Y, t: x_curr if Y[t][0].isupper() else -1,
            K
        ),

        # Is first word?
        (
            lambda x_prev, x_curr, Y, t: x_curr if t == 0 else -1,
            K
        ),

        # Is last word?
        (
            lambda x_prev, x_curr, Y, t: x_curr if t == len(Y) - 1 else -1,
            K
        ),

        # Suffix of one letter
        (
            suffix_1_leters,
            K * len(keywords_1)

         ),

        # Suffix of two letters
        (
            suffix_2_leters,
            K * len(keywords_2)

        ),

        # Suffix of three letters
        (
            suffix_3_leters,
            K * len(keywords_3)

        ),

        # After special words
        (
            lambda x_prev, x_curr, Y, t: x_curr + K*(special_prev_word_dict[Y[t-1]]) if
            (Y[t-1] in special_prev_word_dict and t > 0) else -1,
            K*len(special_prev_word_dict)
        )
    ]

    if current_feature is None:
        features = all_features
    else:
        features = [all_features[0], all_features[1], all_features[current_feature]]

    def phi(x_prev, x_curr, Y, t):
        phi_idxs = []
        block_idx = 0

        for feature, block_length in features:
            idx = feature(x_prev, x_curr, Y, t)
            if idx != -1:
                phi_idxs.append(block_idx + idx)
            block_idx += block_length

        return phi_idxs

    w_len = sum(length for _, length in features)
    w = np.zeros((w_len, 1)) + 1e-12

    return phi, w
