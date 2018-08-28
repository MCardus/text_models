"""Doc2Vec tests"""

import os
import unittest
import logging
import numpy as np
from text_models.embeddeds import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

class TestLDA(unittest.TestCase):

    def setUp(self):
        self.doc2vec = Doc2Vec()
        logging.basicConfig(level=logging.DEBUG)

    def test_preprocessing(self):
        input_text = ["Hello my dear world", "I like sushi from World Japan"]
        tagged_doc = self.doc2vec.pre_process(input_text)
        logging.info(tagged_doc)
        assert isinstance(tagged_doc, list)
        assert isinstance(tagged_doc[0], TaggedDocument)

    def test_fit(self):
        input_text = ["Hello my dear world", "I like sushi from World Japan"]
        output = self.doc2vec.fit(data_list=input_text)
        assert os.path.isfile(self.doc2vec.default_model_pickle_filepath) and \
               os.access(self.doc2vec.default_model_pickle_filepath, os.R_OK)

    def test_predict(self):
        input_text= "Dear world, I like sushi"
        output = self.doc2vec.predict(input_text=input_text)
        assert isinstance(output,np.ndarray)





if __name__ == '__main__':
    unittest.main()
