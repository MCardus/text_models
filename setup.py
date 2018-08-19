from distutils.core import setup

setup(
    name='text_models',
    packages=['topic_modelling/lda', 'utils'],
    version='0.1',
    license='MIT',
    long_description=open('README.md').read()
)
