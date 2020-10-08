import os

from nltk.stem.porter import PorterStemmer as PS

import logging
from logging import getLogger, StreamHandler, Formatter, FileHandler


def init_logger(logfile="./log.txt"):
    logger = getLogger(__name__)

    logger.setLevel(logging.INFO)

    # log to console and to file both
    stream_handler = StreamHandler()
    file_handler = FileHandler(logfile, 'a', encoding='utf-8')

    stream_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    handler_format = Formatter('[%(asctime)s][%(levelname)s] - %(message)s')
    stream_handler.setFormatter(handler_format)
    file_handler.setFormatter(handler_format)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def load_stop_words(stopwords_file=None):
    if stopwords_file is None:
        srcdir = os.path.dirname(os.path.abspath(__file__))
        stopwords_file = os.path.join(srcdir, "stopwords.txt")
    stopwords = []
    with open(stopwords_file, "r") as f:
        for l in f:
            stopwords.append(l.strip())
    return set(stopwords)


def stem(x):
    return PS().stem(x)
