from distutils.core import setup

setup(
    name='text_models',
    packages=['text_models',
              'text_models/topic_modelling',
              'text_models/topic_modelling/lda',
              'text_models/embeddeds',
              'text_models/embeddeds/doc2vec',
              'text_models/utils'],
    version='0.1',
    license='MIT',
    long_description=open('README.md').read()
)
