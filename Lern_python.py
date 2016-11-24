import logging
import sys
import unicodedata
from collections import defaultdict
from os import path

import gensim
import pandas as pd
from nltk.tokenize import RegexpTokenizer


def trainWord2Vec(mypath):
    METADATA_FILE = path.join(mypath + 'metadata.csv')

    # read csv-data (separated by semicolons)
    metadata = pd.read_csv(METADATA_FILE, sep=";", encoding="utf-8")

    # convert nan-values to empty strings
    metadata = metadata.fillna("")

    # use the "id" field as index
    metadata = metadata.set_index("id")

    documents = ((((((metadata.creator + " ").str.cat(metadata.contributor) + " ").str.cat(
        metadata.title) + " ").str.cat(metadata.description) + " ").str.cat(metadata.subject) + " ").str.cat(
        metadata.country).str.strip()).values

    print(documents[2])

    tokenizer = RegexpTokenizer(r'\w+')

    texts = []
    count = 0;

    for document in documents:

        percent = round(100 / len(documents) * count, 2)
        sys.stdout.write('\r')
        sys.stdout.write(str(percent) + '%')
        sys.stdout.flush()
        count += 1;

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
        # remove words that appear only once
        frequency = defaultdict(int)

        for text in texts:
            for token in text:
                frequency[token] += 1

        outfile = open('D:/Dropbox/Dropbox_Uni/Europena/test_list.txt', 'w')
        outfile.write('\n'.join(texts))

        texts = [[token for token in text if (frequency[token] > 1)] for text in texts]

    print('begin training ')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec(texts, size=100, window=10, min_count=5, workers=4)
    model.save(path.join(mypath, 'metadata_100x10x5'))


trainWord2Vec('D:/Dropbox/Dropbox_Uni/Europena/')
