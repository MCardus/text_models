"""LDA Model. The package provides fit, transform and preprocessing services"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
import logging
from utils import setup_logging, timed, read_pickle_file, remove_from_memory, write_pickle_file, tokenize


class LDA(object):

    def __init__(self):
        setup_logging(default_path="utils/logging_properties.yml", severity_level=logging.INFO)
        self.NUM_CONCURRENT_GRIDSEARCH_JOBS = 5
        self.NUM_CONCURRENT_LDA_JOBS = 12
        self.DEFAULT_VECTORIZER_PICKLE_FILEPATH = "vectorizer.pkl"
        self.DEFAULT_LDA_PICKLE_FILEPATH = "lda.pkl"

    @timed
    def pre_process(self, texts_list, vectorizer_file=None, min_freq=0.1, max_freq=0.8,):
        """
        Pre-process raw data and generates a convenient structure to fit model's darta
        :param texts_list: Numpy array of texts to be pre-processed
        :param vectorizer_file: File path pointing to an existing vectorizer. String
        :param min_freq: Min number of word coocurrences for considering a term within the vocab. Integer value
        :param max_freq: Max percentage for considering a term within the vocab. Floating value between 0.-1
        :return: Sparse matrix structure containing the document term matrix (DTM)
        """
        logging.info(f"""Received {len(texts_list)} elements to be pre-processed""")
        # Trying to load an already existing vectorizer
        if vectorizer_file:
            vectorizer = read_pickle_file(file_path=vectorizer_file)
            vectorized_documents = vectorizer.transform(texts_list)

        else:
            # Computing document term matrix
            vectorizer = CountVectorizer(tokenizer=tokenize, min_df=min_freq, max_df=max_freq)
            vectorized_documents = vectorizer.fit_transform(texts_list)
            write_pickle_file(vectorizer, self.DEFAULT_VECTORIZER_PICKLE_FILEPATH)
        dtm_sparsicity = ((vectorized_documents.todense() > 0).sum() /
                          float(vectorized_documents.todense().size)) * 100
        logging.info(f"""DTM sparsisity {dtm_sparsicity}""")
        dtm_vocabulary = vectorizer.vocabulary_
        logging.info(f"""DTM vocabulary {dtm_vocabulary}""")
        # Removing from memory not needed objects
        remove_from_memory(texts_list)
        return vectorized_documents

    @timed
    def fit(self, texts_list, max_topics=60):
        """
        Fit LDA model using gridsearch (hyperparameters optimitzation)
        :param texts_list: Numpy array of texts to be pre-processed
        :param max_topics: Max number of topics
        """
        topic_search_granurality = 10
        # Pre-processing
        dtm = self.pre_process(texts_list)
        # Fit
        gridsearch_params = {
            'n_components': [component for component in range(topic_search_granurality,
                                                              max_topics,
                                                              topic_search_granurality)],
            'learning_decay': [.7]
        }
        logging.info(f"""Gridsearch params {gridsearch_params}""")
        lda_model = LatentDirichletAllocation(n_jobs=self.NUM_CONCURRENT_LDA_JOBS)
        grid_search = GridSearchCV(lda_model, param_grid=gridsearch_params, n_jobs=self.NUM_CONCURRENT_GRIDSEARCH_JOBS)
        grid_search.fit(dtm)
        best_lda_model = grid_search.best_estimator_
        logging.info(f"""Best Model's Params {best_lda_model}""")
        logging.info(f"""Best Log Likelihood Score {grid_search.best_score_}""")
        logging.info(f"""Model Perplexity {best_lda_model.perplexity(dtm)}""")
        # Serializating results
        write_pickle_file(grid_search, self.DEFAULT_LDA_PICKLE_FILEPATH)

    @timed
    def predict(self, text):
        dtm = self.pre_process(text, vectorizer_file=self.DEFAULT_VECTORIZER_PICKLE_FILEPATH)
        lda_model = read_pickle_file(self.DEFAULT_LDA_PICKLE_FILEPATH)
        if dtm==None or lda_model==None:
            raise ValueError("Can't find vectorizer or lda models loaded, please call fit lda method again")
        return lda_model.transform(dtm)
