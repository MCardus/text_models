"""Word2Vec Model. The package provides fit, transform and prepocessing services"""

import logging
from gensim.models import Word2Vec as Word2VecGensim
from text_models.utils import setup_logging, tokenize
import numpy as np

class Word2vec(object):

    def __init__(self, default_model_pickle_filepath="w2v.model"):
        setup_logging(default_path="utils/logging_properties.yml", severity_level=logging.INFO)
        self.default_model_pickle_filepath = default_model_pickle_filepath

    def pre_process(self, data_list):
        logging.info(f"""Applying word2vc pre_process to {len(data_list)} documents""")
        return [tokenize(doc) for doc in data_list]

    def fit(self, data_list, max_epochs=20, alpha=0.025, min_alpha=0.0000025, min_count_freq=0.0001, min_count=1, dm=1):
        logging.info(f"""Word2vec fit using max_epochs {max_epochs}""")
        documents = self.pre_process(data_list)
        model = Word2VecGensim(
                        sentences=documents,
                        size=300,
                        alpha=alpha,
                        min_alpha=0.00025,
                        min_count=1)
        model.train(documents, total_examples=len(documents), epochs=max_epochs)

        model.save(self.default_model_pickle_filepath)
        logging.info("Model Saved")



    def predict(self, input_text):
        model = Word2VecGensim.load(self.default_model_pickle_filepath)

        doc_vector = np.array([model[token] for token in tokenize(input_text) if token in model])
        logging.info(f"""word2vec vector: {doc_vector}""")
        return doc_vector

