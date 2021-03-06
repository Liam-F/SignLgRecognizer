#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
An example code to test using unittest

@author: udacity

Created on MM/DD/YYYY
"""

from aind.asl_data import SinglesData, WordsData
import numpy as np
from IPython.core.display import display, HTML

RAW_FEATURES = ['left-x', 'left-y', 'right-x', 'right-y']
GROUND_FEATURES = ['grnd-rx', 'grnd-ry', 'grnd-lx', 'grnd-ly']


def get_error(guesses: list, test_set: SinglesData):
    '''return the WER of the guesses, the total of correct words and the total
    number of words'''
    S = 0
    N = len(test_set.wordlist)
    num_test_words = len(test_set.wordlist)
    if len(guesses) != num_test_words:
        s_txt = "Size of guesses must equal number of test words ({})!"
        print(s_txt.format(num_test_words))
    for word_id in range(num_test_words):
        if guesses[word_id] != test_set.wordlist[word_id]:
            S += 1
    return float(S) / float(N), N - S, N


def show_errors(guesses: list, test_set: SinglesData):
    """ Print WER and sentence differences in tabular form

    :param guesses: list of test item answers, ordered
    :param test_set: SinglesData object
    :return:
        nothing returned, prints error report

    WER = (S+I+D)/N  but we have no insertions or deletions for isolated words
    so WER = S/N
    """
    S = 0
    N = len(test_set.wordlist)
    num_test_words = len(test_set.wordlist)
    if len(guesses) != num_test_words:
        s_txt = "Size of guesses must equal number of test words ({})!"
        print(s_txt.format(num_test_words))
    for word_id in range(num_test_words):
        if guesses[word_id] != test_set.wordlist[word_id]:
            S += 1

    print("\n**** WER = {}".format(float(S) / float(N)))
    print("Total correct: {} out of {}".format(N - S, N))
    print('Video  Recognized' + ' ' * 52 + 'Correct')
    print('=' * 95)
    for video_num in test_set.sentences_index:
        correct_sentence = [test_set.wordlist[i] for i in
                            test_set.sentences_index[video_num]]
        recognized_sentence = [guesses[i] for i in
                               test_set.sentences_index[video_num]]
        for i in range(len(recognized_sentence)):
            if recognized_sentence[i] != correct_sentence[i]:
                recognized_sentence[i] = '*' + recognized_sentence[i]
        s_txt1 = ' '.join(recognized_sentence)
        s_txt2 = ' '.join(correct_sentence)
        print('{:5}: {:60}  {}'.format(video_num, s_txt1, s_txt2))


def getKey(item):
    return item[1]


def train_all_words(training: WordsData, model_selector):
    """ train all words given a training set and selector

    :param training: WordsData object (training set)
    :param model_selector: class (subclassed from ModelSelector)
    :return: dict of models keyed by word
    """
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word,
                               n_constant=3).select()
        model_dict[word] = model
    return model_dict


def combine_sequences(split_index_list, sequences):
    '''
    concatenate sequences referenced in an index list and returns tuple of the
    new X,lengths

    useful when recombining sequences split using KFold for hmmlearn

    :param split_index_list: a list of indices as created by KFold splitting
    :param sequences: list of feature sequences
    :return: tuple of list, list in format of X,lengths use in hmmlearn
    '''
    sequences_fold = [sequences[idx] for idx in split_index_list]
    X = [item for sublist in sequences_fold for item in sublist]
    lengths = [len(sublist) for sublist in sequences_fold]
    return X, lengths


def putHTML(color, msg):
    source = """<font color={}>{}</font><br/>""".format(color, msg)
    return HTML(source)


def feedback(passed, failmsg='', passmsg='Correct!'):
    if passed:
        return putHTML('green', passmsg)
    else:
        return putHTML('red', failmsg)


def test_features_tryit(asl):
    print('asl.df sample')
    display(asl.df.head())
    sample = asl.df.ix[98, 1][GROUND_FEATURES].tolist()
    correct = [9, 113, -12, 119]
    s_txt = 'The values returned were not correct.  Expected: {} Found: {}'
    failmsg = s_txt.format(correct, sample)
    return feedback(sample == correct, failmsg)


def test_std_tryit(df_std):
    print('df_std')
    display(df_std)
    sample = df_std.ix['man-1'][RAW_FEATURES]
    correct = [15.154425, 36.328485, 18.901917, 54.902340]
    s_txt = 'The raw man-1 values returned were not correct.\nExpected: {} '
    s_txt += 'for {}'
    failmsg = s_txt.format(correct, RAW_FEATURES)
    return feedback(np.allclose(sample, correct, .001), failmsg)
