#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
An example code to test using unittest

@author: udacity, ucaiado

Created on 09/24/2017
"""

import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from aind.asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict,
                 this_word: str, n_constant=3, min_n_components=2,
                 max_n_components=10, random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states,
                                    covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state,
                                    verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                s_err = "model created for {} with {} states"
                print(s_err.format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                s_err = "failure on {} with {} states"
                print(s_err.format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        raise NotImplementedError


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application
    to hmm topology optimization." Document Analysis and Recognition, 2003.
    Proceedings. Seventh International Conference on. IEEE, 2003.
    https://goo.gl/CkgJ4H
    https://goo.gl/Za9QWE
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        raise NotImplementedError


class SelectorCV(ModelSelector):
    '''
    select best model based on average log Likelihood of cross-validation
    folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # implement model selection using CV
        split_method = KFold(min(3, len(self.sequences)))
        l_aux = range(self.min_n_components, self.max_n_components + 1)
        best_avg = -10e8
        best_model = self.base_model(self.n_constant)
        i_seq = self.sequences
        for num_states in l_aux:
            curr_err = 0.
            i_count = 0
            try:
                iter_obj = split_method.split(self.sequences)
                for train_idx, test_idx in iter_obj:
                    X_train, Y_train = combine_sequences(train_idx, i_seq)
                    X_test, Y_test = combine_sequences(test_idx, i_seq)
                    hmm_model = GaussianHMM(n_components=num_states,
                                            covariance_type='diag',
                                            n_iter=1000,
                                            random_state=self.random_state,
                                            verbose=False)
                    hmm_model.fit(X_train, Y_train)
                    curr_err += hmm_model.score(X_test, Y_test)
                    i_count += 1
            except ValueError:
                continue
                # print(i_count)
            # select the model with the smaller error
            if i_count == 0:
                best_model = hmm_model
            elif curr_err/i_count > best_avg:
                best_avg = curr_err/i_count
                best_model = hmm_model
        return best_model
