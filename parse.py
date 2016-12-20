from __future__ import division
import sys
import gzip
import random


S_END = '.'
IGNORE = ["'", " ", "``", ""]


def sentence_iterator(gzip_file, max_sentences=sys.maxsize):
    x, y = [], []
    i = 0
    with gzip.GzipFile(gzip_file) as input_file:
        while True:
            pair = input_file.readline()
            if not pair:
                break
            pair = pair.decode('utf-8').strip().split(' ')
            if pair[0] in IGNORE:
                continue
            if pair[0] == S_END and len(x) > 1:
                yield x, y
                i += 1
                if i > max_sentences:
                    break
                x, y = [], []
                input_file.readline()  # skip empty line
            else:
                x.append(pair[0])
                y.append(pair[1])


def collect_sets(gzip_filename, k=2, n=1, max_sentences=sys.maxsize):
    """
    returns a collection of test and train sets as requested. If only one set is requested (n=1)
    then the set list is redundant and you can access the single set directly. See below.

    :param gzip_filename: file name of gzipped data
    :param k: cross-validation factor, if set to e.g. 4, size of test set is 1/4 of data
    :param n: number of train/test sets to draw
    :param max_sentences: maximal number of sentences to read from file
    :return:
        sets - a collection of n pairs of train/test collections
        xvals - a set of the values of x (parts of speech)
        yvals - a set of the values of y (words)
    usage:
        data, xv, yv = collect_sets('data_split.gz', k=10, n=1) #9/10 of data is for training, only one copy
        # data['train'] - list of sentences
        # data['train'][0][0] - a list of POS tags for the first sentence
        # data['train'][0][1] - corresponding list of words comprising the first sentence
        # data['train'][1][0] - a list of POS tags for the second sentence
        # data['train'][1][1][4] - the 5th word in the 2nd sentence of the train set
        # data['train'][1][1][1][4] - (If n>=2) the 5th word in the 2nd sentence of the 2nd training set
    """

    sets = [{'train': [], 'test': []} for _ in range(n)]
    xvals, yvals = set([]), set([])
    for sentence in sentence_iterator(gzip_filename, max_sentences=max_sentences):
        for i in range(len(sentence[0])):
            xvals.add(sentence[0][i])
            yvals.add(sentence[1][i])
        for ni in range(n):
            if random.random() < 1/k:
                sets[ni]['test'].append(sentence)
            else:
                sets[ni]['train'].append(sentence)
    if n == 1:
        sets = sets[0]
    return sets, xvals, yvals
