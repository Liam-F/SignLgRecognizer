#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement a recognizer that used models already trained to classify unseen data

@author: udacity, ucaiado

Created on 10/03/2017
"""

import warnings
from aind.asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object,
       'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses. both lists are ordered by
       the test set word_id probabilities is a list of dictionaries where each
       key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set
       word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # implement the recognizer
    Xlengths = test_set.get_all_Xlengths()
    for X_test, y_test in Xlengths.values():
        best_guess = None
        best_prob = -1e8
        d_guesses = {}
        for word, hmm_model in models.items():
            try:
                # if the probability to be this word is greater
                logL = hmm_model.score(X_test, y_test)
                d_guesses[word] = logL
                if logL > best_prob:
                    # it is the best guess
                    best_guess = word
                    best_prob = logL
            except (ValueError, AttributeError) as e:
                d_guesses[word] = None
        probabilities.append(d_guesses)
        guesses.append(best_guess)

    return probabilities, guesses
