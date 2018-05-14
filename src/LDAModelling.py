"""LDA Model. This class can preprocess data, train and transform"""
import json
import pandas as pd
import numpy as np
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
import logging
from sklearn.externals import joblib
import gc
from functools import wraps
import logging
import time


class LDAModelling(object):

    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)

    def timed(func):
        """This decorator prints the execution time for the decorated function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            logging.info("{} ran in {}s".format(func.__name__, round(end - start, 2)))
            return result
        return wrapper

    @timed
    def _pre_process(self, dataset, min_freq=10, max_freq=0.8):
        """
        Pre process raw data
        :param dataset: Input dataset. CSV containing fields at least title and content fields
        :param min_freq: Min number of ocurrences for considering a term within the vocab. Integer value
        :param max_freq: Max percentage for considering a term within the vocab. Floating value
        :return: Sparse matrix containing the document term matrix (DTM)
        """
        # Reading raw data
        raw_data = pd.read_csv(dataset)
        logging.info(json.dumps({"Service": "_pre_process", "Input data shape" : raw_data.shape}))
        # Selecting needed fields
        data = raw_data["content"].values
        # Computing document term matrix
        vectorizer = CountVectorizer(tokenizer=self._tokenize, min_df=min_freq, max_df=max_freq)
        vectorized_documents = vectorizer.fit_transform(data)
        dtm_sparsicity = ((vectorized_documents.todense() > 0).sum() / float(vectorized_documents.todense().size)) * 100
        dtm_vocabulary = vectorizer.vocabulary_
        logging.info(json.dumps({"Service": "_pre_process", "DTM sparsisity": dtm_sparsicity}))
        logging.info(json.dumps({"Service": "_pre_process", "DTM vocabulary": dtm_vocabulary}))
        # Removing from memory not needed objects
        #del raw_data
        #del data
        #gc.collect()
        return vectorized_documents

    @timed
    def train(self, dataset, num_topics=90):
        """
        Train LDA model.
        :param dataset: Input dataset. CSV containing fields at least title and content fields
        :param num_topics: Max number of topics
        :return: Void but it serialize the model into a pikle format
        """
        topic_search_granurality = 10
        document_term_matrix = self._pre_process(dataset)
        search_params = {'n_components': [component for component in xrange(topic_search_granurality,
                                                                            num_topics,
                                                                            topic_search_granurality)],
                         'learning_decay': [.7]}
        logging.info(json.dumps({"Service": "train", "Search params": search_params}))
        lda_model = LatentDirichletAllocation(n_jobs=4)
        grid_search = GridSearchCV(lda_model, param_grid=search_params, n_jobs=4)
        grid_search.fit(document_term_matrix)
        best_lda_model = grid_search.best_estimator_

        logging.info(json.dumps({"Service": "train", "Best Model's Params": grid_search.best_params_}))
        logging.info(json.dumps({"Service": "train", "Best Log Likelihood Score": grid_search.best_score_}))
        logging.info(json.dumps({"Service": "train",
                                 "Model Perplexity": best_lda_model.perplexity(document_term_matrix)}))

        joblib.dump(best_lda_model, 'output_model.pkl')

    @timed
    def transform(self):
        pass

    def _tokenize(self,text):
        """
        English tokenitzation function
        Applied filters:
        * Min length words filter
        * Lower case transformation
        * Literals filter (no numbers either punctuation allows)
        * Stem filter (PorterStemmer to remove morphological affixes)
        * Stop words filter
        @TODO - lemmaritzxation
        @PoS Tagging
        """
        min_length = 3
        words = map(lambda word: word.lower(), word_tokenize(text))
        words = [word for word in words
                 if word not in stopwords.words("english")]
        tokens = (list(map(lambda token: PorterStemmer().stem(token),
                           words)))
        p = re.compile('[a-zA-Z]+')
        filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= min_length, tokens))
        return filtered_tokens


if __name__ == "__main__":
    lda = LDAModelling()
    lda.train("articles1_small.csv")

