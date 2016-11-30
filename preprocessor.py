import pandas as pd
import sys
import unicodedata
import logging

from os import path
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from stop_words import get_stop_words

mypath = ''
METADATA_FILE = ''
metadata = ''


def __init__(self, mypath):
    self.mypath = mypath
    METADATA_FILE = path.join(mypath + 'TEST_translation.csv')
    # read csv-data (separated by semicolons)
    self.metadata = pd.read_csv(METADATA_FILE, sep=";", encoding="utf-8")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    texts = preprocess(metadata)
    return texts


def remove_single_words(texts):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('removing words that appear only once')

    # remove words that appear only once
    frequency = defaultdict(int)

    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if (frequency[token] > 1)] for text in texts]
    return texts


def preprocess(metadata):
    logging.info('starting preprocessing metadata')

    # convert nan-values to empty strings
    metadata = metadata.fillna("")

    # use the "id" field as index
    metadata = metadata.set_index("id")

    # concatinate to string:
    #   - (id)
    #   - contributor
    #   - creator
    #   - date
    #   - describtion
    #   - spatial
    #   - subject
    #   - type
    #   - year
    documents = ((((((((metadata.contributor + " ").str.cat(metadata.creator) + " ").str.cat(
        metadata.date) + " ").str.cat(metadata.description) + " ").str.cat(metadata.spatial) + " ").str.cat(
        metadata.subject + " ").str.cat(metadata.type) + " ").str.cat(metadata.year) + " ").str.strip()).values

    stop_words = []
    stop_words.extend(get_stop_words('en'))
    stop_words.extend(get_stop_words('de'))
    stop_words.extend(get_stop_words('fr'))
    stop_words.extend(get_stop_words('it'))
    stop_words.extend(get_stop_words('pt'))
    stop_words.extend(get_stop_words('ro'))
    stop_words.extend(get_stop_words('spanish'))

    tokenizer = RegexpTokenizer(r'\w+')

    texts = []
    count = 1;

    for document in documents:

        ####################################### progress in percent
        percent = round(100 / len(documents) * count, 2)
        sys.stdout.write('\r')
        sys.stdout.write(str(percent) + '%')
        sys.stdout.flush()
        count += 1;
        ###########################################################

        valid_words = []

        # skip empty documents
        if len(document) > 0:

            # for each lower-case transformed word
            for word in tokenizer.tokenize(document.lower()):
                # remove surrounding whitespace and line endings
                word = word.strip()

                # normalize, remove accents and umlaute
                word = unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore')
                valid_words.append(word)

        texts.append(valid_words)
    print('\n')

    texts = remove_single_words(texts)
    return texts

    # outfile = open(path.join(mypath, 'test_list.txt'), 'w')
    # outfile.write('\n'.join(texts))