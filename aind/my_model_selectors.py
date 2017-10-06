#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement different model selection strategies to Hiden Markov Models

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

    Wiki: https://en.wikipedia.org/wiki/Hidden_Markov_model#Architecture
    Docs: https://goo.gl/gE2JYK
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on BIC scores
        l_aux = range(self.min_n_components, self.max_n_components + 1)
        best_bic = 10e8
        best_model = self.base_model(self.n_constant)
        n, f = self.X.shape
        for num_states in l_aux:
            curr_err = 10e8
            try:
                hmm_model = self.base_model(num_states)
                # logL is the score of the model
                curr_err = hmm_model.score(self.X, self.lengths)

                # ###### UDACITY CODE REVIEWR SUGESTION ################
                # "Free parameters" are parameters that are learned by the
                # model and it is a sum of:
                # 1. The free transition probability parameters, which is the
                # size of the transmat matrix less one row because they add up
                # to 1 and therefore the final row is deterministic, so
                # `n*(n-1)`
                i_transmat = num_states * (num_states-1)

                # 2. The free starting probabilities, which is the size of
                # startprob minus 1 because it adds to 1.0 and last one can be
                # calculated so `n-1`
                i_startprob = num_states - 1

                # 3. Number of means, which is `n*f`, where `f` is the number
                # of features
                i_emission = num_states * f

                # 4. Number of covariances which is the size of the covars
                # matrix, which for "diag" is `n*f`
                i_emission += num_states * f

                # ###### OLD DEFINITION ################
                # calculate the Free parameters
                # (1) The transition probabilities is a N × N matrix and is
                # called Markov matrix. Because any one transition probability
                # can be determined once the others are known, there are a
                # total of N(N-1) transition parameters (Wiki)
                # i_transmat = num_states * (num_states-1)

                # (2) The emission probabilities to GaussianHMM is the number
                # of free size of means and covar matrix, where the last one is
                # diagonal
                # i_emission = hmm_model.means_.shape[0]
                # i_emission *= hmm_model.means_.shape[1]
                # i_emission += sum(sum(sum(hmm_model.covars_ != 0)))

                # (3) Start probability attr, has the size of the # of states
                # i_startprob = num_states
                # p = size(transmat) + size(emission) + size(initial)
                n_freeparm = i_transmat + i_emission + i_startprob

                # BIC = -2 * logL + p * logN
                curr_bic = -2 * curr_err + n_freeparm * np.log(self.X.shape[0])
            except (ValueError, AttributeError) as e:
                continue
            # select the model with the smaller error
            if curr_bic < best_bic:
                best_bic = curr_bic
                best_model = hmm_model
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))

    Biem, Alain. "A model selection criterion for classification: Application
    to hmm topology optimization." Document Analysis and Recognition, 2003.
    Proceedings. Seventh International Conference on. IEEE, 2003.
    link: https://goo.gl/CkgJ4H
    '''

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    def select(self):
        # Udacity code reviers sugestion
        best_model = None
        best_score = float("-Inf")
        for n in range(self.min_n_components, self.max_n_components + 1):
            # measure the error of all "competing classes"
            try:
                scores = []
                hmm_model = self.base_model(n)
                f_logL = float("-Inf")
                for w, (X, lengths) in self.hwords.items():
                    if w != self.this_word:
                        # Please develop the rest of your function below
                        scores.append(hmm_model.score(X, lengths))
                    else:
                        f_logL = hmm_model.score(X, lengths)
            except (ValueError, AttributeError) as e:
                continue
            # the Discriminant Factor Criterion is the difference between the
            # evidence of the model, given the corresponding data set, and the
            # average over anti-evidences of the model
            m = len(scores) * 1.
            f_sumLogL = sum(scores)
            curr_dic = f_logL - (f_sumLogL - f_logL)/(m-1.)
            if curr_dic > best_score:
                best_score = curr_dic
                best_model = hmm_model
        return best_model

    def select_OLD(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on DIC scores
        l_aux = range(self.min_n_components, self.max_n_components + 1)
        best_dic = -10e8
        best_nsatets = self.n_constant
        # measure the error of all "competing classes" using the same data set
        d_competing_errs = {}
        for num_states in l_aux:
            curr_err = 10e8
            try:
                hmm_model = self.base_model(num_states)
                # logL is the score of the model
                curr_err = hmm_model.score(self.X, self.lengths)
                d_competing_errs[num_states] = curr_err
            except (ValueError, AttributeError) as e:
                continue

        # the Discriminant Factor Criterion is the difference between the
        # evidence of the model, given the corresponding data set, and the
        # average over anti-evidences of the model
        f_sumLogL = sum(d_competing_errs.values())  # SUM(log(P(X(all))
        m = len(d_competing_errs) * 1.
        for num_states in d_competing_errs.keys():
            # measure DIC, the "evidence"
            # DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
            f_logL = d_competing_errs[num_states]
            curr_dic = f_logL - (f_sumLogL - f_logL)/(m-1.)
            # choose a model which maximizes the evidence
            if curr_dic > best_dic:
                best_dic = curr_dic
                best_nsatets = num_states
        return self.base_model(best_nsatets)


class SelectorCV(ModelSelector):
    '''
    select best model based on average log Likelihood of cross-validation
    folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # implement model selection using CV
        split_method = KFold()
        l_aux = range(self.min_n_components, self.max_n_components + 1)
        best_avg = -10e8
        best_nstates = self.n_constant
        X_all = self.sequences
        for num_states in l_aux:
            curr_err = 0.
            i_count = 0
            try:
                iter_obj = split_method.split(self.sequences)
                for train_idx, test_idx in iter_obj:
                    X_train, Y_train = combine_sequences(train_idx, X_all)
                    X_test, Y_test = combine_sequences(test_idx, X_all)
                    hmm_model = GaussianHMM(n_components=num_states,
                                            covariance_type='diag',
                                            n_iter=1000,
                                            random_state=self.random_state,
                                            verbose=False)
                    hmm_model.fit(X_train, Y_train)
                    curr_err += hmm_model.score(X_test, Y_test)
                    i_count += 1
            except (ValueError, AttributeError) as e:
                continue
                # print(i_count)
            # select the model with the smaller error
            if i_count == 0:
                best_nstates = num_states
            elif curr_err/i_count > best_avg:
                best_avg = curr_err/i_count
                best_nstates = num_states
        return self.base_model(best_nstates)
