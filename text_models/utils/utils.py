"""Text models utilities"""

from functools import wraps
import time
import os
import logging
import logging.config
import yaml
import pickle
import gc
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords


def tokenize(text, token_min_length=3):
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
    :param text: Input text to be tokenized
    :param token_min_length: Token min length. Integer
    """
    words = map(lambda word: word.lower(), word_tokenize(text))
    words = [word for word in words
             if word not in stopwords.words("english")]
    tokens = (list(map(lambda token: PorterStemmer().stem(token),
                       words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(token) and len(token) >= token_min_length, tokens))
    return filtered_tokens


def timed(func):
    """Decorator to print execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info(f"""{func.__name__} ran in {round(end - start, 2)}s""")
        return result

    return wrapper


def setup_logging(
        default_path='logging_properties.yml',
        severity_level=logging.DEBUG,
        env_key='LOG_CFG'):
    """Setup logging configuration"""

    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    path = os.path.join(os.getcwd(), path)
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.load(f)
        logging.config.dictConfig(config)
        logging.getLogger().setLevel(severity_level)
    else:
        logging.basicConfig(level=severity_level)


def read_pickle_file(file_path):
    """
    Read pickle file
    :param file_path: String pointing to a pickle file
    :return: Pickle object
    """
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

def write_pickle_file(model, output_file_path):
    """
    Write pickle file
    :param: model to be serialized. An object.
    :param file_path: Serialitzation filepath. String
    """
    with open(output_file_path, 'wb') as f:
        pickle.dump(model, f)


def remove_from_memory(var):
    """Remove a certain variable from memory"""
    del var
    gc.collect()
