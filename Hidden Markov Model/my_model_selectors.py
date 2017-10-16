import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
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
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
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
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score, best_model = float("inf"), None

        # iterate through the num_states to find the best model
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(n_components)
                logL = hmm_model.score(self.X, self.lengths)
                n_features = sum(self.lengths)
                n_params = n_components**2 + 2*n_components*n_features - 1
                logN = np.log(n_features)
                bic = -2*logL + n_params*logN
                if bic < best_score:
                    best_score, best_model = bic, hmm_model
            except:
                pass

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def generate_words(self):
        """ iterates through the words but leaves out the word that belongs to the model
        """
        return [self.hwords[word] for word in self.words if word != self.this_word]
    
    def logL_of_words(self, model, words):
        """ finds the score of the 
        """
        return [model.score(word[0], word[1]) for word in words]

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score, best_model, models, words = float("-inf"), None, [], []
        
        # generate the words
        words = self.generate_words()
        
        # iterates through num_states to create models
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(n_components)
                logL = hmm_model.score(self.X, self.lengths)
                models.append((hmm_model, logL))
            except:
                pass
        
        for model in models:
            hmm_model, logL = model
            dic = logL - np.mean(self.logL_of_words(hmm_model, words))
            if dic > best_score:
                best_score, best_model = dic, hmm_model
        
        return best_model
            

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        n_splits = 2
        best_score, best_model = float("-inf"), None
        kf = KFold(n_splits=n_splits)

        # iterate through the number of states to see which one is best
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            if len(self.sequences) < n_splits:
                break
            
            scores, hmm_model = [], None

            # iterate through the n_split folds of sequences
            for train, test in kf.split(self.sequences):
                x_train, lengths_train = combine_sequences(train, self.sequences)
                x_test, lengths_test = combine_sequences(test, self.sequences)

                # build the model with the training data and score with test
                try: 
                    self.X = x_train
                    self.lengths = lengths_train
                    hmm_model = self.base_model(n_components)
                    logL = hmm_model.score(x_test, lengths_test)
                    scores.append(logL)
                except:
                    pass
            # picks the best average and best model
            logL_avg = np.average(scores) if len(scores) > 0 else float("inf")
            if logL_avg > best_score:
                best_score, best_model = logL_avg, hmm_model
        
        return best_model
