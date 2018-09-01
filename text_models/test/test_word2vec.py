"""Doc2Vec tests"""

import os
import unittest
import logging
import numpy as np
from text_models.embeddeds import Word2vec

class TestWord2vec(unittest.TestCase):

    def setUp(self):
        self.word2vec = Word2vec()
        logging.basicConfig(level=logging.DEBUG)

    def test_preprocessing(self):
        input_text = ["Hello my dear world"]
        tokens = self.word2vec.pre_process(input_text)
        logging.info(tokens)
        assert isinstance(tokens, list)

    def test_fit(self):
        input_text = ["Hello my dear world", "I like sushi from World Japan"]
        output = self.word2vec.fit(data_list=input_text)

    def test_predict(self):
        input_text = "Dear world, I like sushi"
        output = self.word2vec.predict(input_text=input_text)
        assert isinstance(output,np.ndarray)





if __name__ == '__main__':
    unittest.main()
