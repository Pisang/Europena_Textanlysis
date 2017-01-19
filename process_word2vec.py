import sys
import unicodedata
from collections import defaultdict
from os import path

import gensim
import logging
import pandas as pd
from gensim import corpora
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words


def trainWord2Vec(mypath):
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    METADATA_FILE = path.join(mypath + 'metadata_merged.csv')

    # read csv-data (separated by semicolons)
    metadata = pd.read_csv(METADATA_FILE, sep=";", encoding="utf-8")

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
    tokenizer = RegexpTokenizer(r'\w+')


    texts = []
    count = 0;

    for document in documents:

        ####################################### progress in percent
        percent = round(100 / len(documents) * count, 2)
        sys.stdout.write('\r')
        if count == len(documents):
            sys.stdout.write(str(percent) + '%\n')
        else:
            sys.stdout.write(str(percent) + '%')
        sys.stdout.flush()
        count += 1;
        ###########################################################

        valid_words = []

        # skip empty documents
        if len(document) > 0:

            # for each lower-case transformed word
            for word in tokenizer.tokenize(document.lower()):
                if word in stop_words or (len(word) <= 1):
                    continue

                # remove surrounding whitespace and line endings
                word = word.strip()

                # normalize, remove accents and umlaute
                word = unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore')
                valid_words.append(word)
        texts.append(str(valid_words))

    print(texts)
    # texts should look like:    sentences = [['first', 'sentence'], ['second', 'sentence']]
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec(texts, size=1000, window=5, min_count=2, workers=4)
    model.save(path.join(mypath + 'original_word2vec_model'))



    # remove words that appear only once
    # frequency = defaultdict(int)
    # for text in texts:
    #     for token in text:
    #         frequency[token] += 1
    #
    # texts = [[token for token in text if (frequency[token] > 1)] for text in texts]
    # texts = str(texts)




    #print(texts)

    #logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # model = gensim.models.Word2Vec(texts, size=1000, window=20, min_count=3, workers=4)
    #model.build_vocab(texts)
    #model.train(texts)

    #model = gensim.models.Word2Vec(texts, min_count=10, workers=4)
    #model.build_vocab(texts)  # can be a non-repeatable, 1-pass generator
    #model.train(texts)  # can be a non-repeatable, 1-pass generator

    # model.save(path.join(mypath + 'original_word2vec_model'))



'''
    print('begin training ')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec(texts, size=100, window=10, min_count=5, workers=4)
    model.save(path.join(mypath, 'metadata_100x10x5'))
'''
trainWord2Vec('D:/Dropbox/Dropbox_Uni/Europena/')