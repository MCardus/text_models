"""Doc2Vec Model. The package provides fit, transform and prepocessing services"""

import logging
from gensim.models.doc2vec import Doc2Vec as Doc2VecGensim, TaggedDocument
from text_models.utils import setup_logging, tokenize

class Doc2Vec(object):

    def __init__(self, default_model_pickle_filepath="d2v.model"):
        setup_logging(default_path="utils/logging_properties.yml", severity_level=logging.INFO)
        self.default_model_pickle_filepath = default_model_pickle_filepath

    def pre_process(self, data_list):
        logging.info(f"""Applying pre_process to {len(data_list)} documents""")
        return [TaggedDocument(words=tokenize(_d.lower()),
                               tags=[str(i)]) for i, _d in enumerate(data_list)]

    def fit(self, data_list, max_epochs=50, alpha=0.025, min_alpha=0.0000025, min_count=1, dm=1):
        logging.info(f"""Doc2Vec fit using max_epochs {max_epochs}""")
        model = Doc2VecGensim(size=300,
                        alpha=alpha,
                        min_alpha=0.00025,
                        min_count=1,
                        dm=1)
        tagged_data = self.pre_process(data_list)
        model.build_vocab(tagged_data)
        for epoch in range(max_epochs):
            logging.info('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.iter)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha

        model.save(self.default_model_pickle_filepath)
        logging.info("Model Saved")



    def predict(self, input_text):
        model = Doc2VecGensim.load(self.default_model_pickle_filepath)
        # to find the vector of a document which is not in training data
        test_data = tokenize(input_text)
        v1 = model.infer_vector(test_data)
        logging.info(f"""doc2vec infer {v1}""")
        return v1
