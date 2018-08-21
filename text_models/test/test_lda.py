"""LDA tests"""

import numpy as np
import unittest
from text_models.topic_modelling import LDA
import logging
import os
from text_models.utils import read_pickle_file
from sklearn.model_selection import GridSearchCV as gridSearchType


class TestLDA(unittest.TestCase):

    def setUp(self):
        self.lda = LDA()
        logging.basicConfig(level=logging.DEBUG)

    def test_preprocessing(self):
        sample_text_list = np.array(["Hello LDA", "LDA rocks!!", "Hello, I love LDA"])
        output = self.lda.pre_process(sample_text_list)
        logging.info(output.todense())
        # Vocab should contain: Hello, rocks and love. Other are discarted
        assert output._shape == (3, 3)
        assert os.path.isfile(self.lda.default_vectorizer_pickle_filepath) == True

    def test_fit(self):
        sample_text_list = np.array(["Hello LDA", "LDA rocks!!", "Hello, I love LDA"])
        self.lda.fit(sample_text_list)
        assert os.path.isfile(self.lda.default_lda_pickle_filepath) == True
        lda_model = read_pickle_file(self.lda.default_lda_pickle_filepath)
        assert type(lda_model) == gridSearchType

    def test_predict(self):
        sample_text_list = np.array(["I rock LDA"])
        output = self.lda.predict(sample_text_list)
        lda_model = read_pickle_file(self.lda.default_lda_pickle_filepath)
        optimal_num_topics = lda_model.best_estimator_._n_components
        logging.info(f"""Output: {output}""")
        assert output.shape[1] == optimal_num_topics



if __name__ == '__main__':
    unittest.main()
