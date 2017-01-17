import logging
import os
import sys
import unicodedata
from collections import defaultdict
from os import path

import pandas as pd
from gensim import corpora, models
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words


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


def loadDocument(mypath):
    METADATA_FILE = path.join(mypath + 'metadata_merged.csv')
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

    # pprint(documents)

    stop_words = []
    stop_words.extend(get_stop_words('en'))

    tokenizer = RegexpTokenizer(r'\w+')

    texts = []
    count = 1;

    logging.info('starting tokenizing')

    for document in documents:

        ####################################### progress in percent
        percent = round(100 / len(documents) * count, 2)
        sys.stdout.write('\r')
        if count == len(documents):
            sys.stdout.write(str(percent) + '%\n')
        else: sys.stdout.write(str(percent) + '%')
        sys.stdout.flush()
        count += 1;
        ###########################################################

        valid_words = []

        # skip empty documents
        if len(document) > 0:

            # for each lower-case transformed word
            for word in tokenizer.tokenize(document.lower()):
                if word in stop_words:
                    continue
                # remove surrounding whitespace and line endings
                word = word.strip()

                # normalize, remove accents and umlaute
                word = unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore')
                valid_words.append(word)

        texts.append(valid_words)
    # end for document in documents

    texts = remove_single_words(texts)

    dictionary = corpora.Dictionary(texts)
    dictionary.save(
        path.join(mypath + 'tutorial/original_dictionary.dict'))  # store the dictionary, for future reference
    # print(dictionary)
    # Dictionary(71 unique tokens: ['comic', 'berlin', 'abraham', 'quot', 'romania']...)
    print(dictionary.token2id)
    # {'comic': 14, 'berlin': 58, 'abraham': 37, ...}

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(path.join(mypath + 'tutorial/original_corpus.mm'), corpus)
    # print(corpus)




def tfIdf_transform(mypath):
    print('\n### tfidf transform ###\n')
    if (os.path.exists(path.join(mypath + 'tutorial/original_dictionary.dict'))):
        dictionary = corpora.Dictionary.load(path.join(mypath + 'tutorial/original_dictionary.dict'))
        corpus = corpora.MmCorpus(path.join(mypath + 'tutorial/original_corpus.mm'))
        print("Used files generated from first tutorial.")
    else:
        print("Please run first tutorial to generate data set.")

    # train the model by going through the supplied corpus once and computing document frequencies of all its features
    tfidf = models.TfidfModel(corpus)  # train the model

    # From now on, tfidf is treated as a read-only object that can be used to convert any vector from the old representation
    # (bag-of-words integer counts) to the new representation (TfIdf real-valued weights)
    corpus_tfidf = tfidf[corpus]
    corpora.MmCorpus.serialize(path.join(mypath + 'tutorial/corpus_tfidf.mm'), corpus)
    # corpus_tfidf is just a wrapper which converts a document if called - transforming the whole thing would contradict
    # gensim's objective of memory independence.

    # for doc in corpus_tfidf:
    # print(doc)

    # If you will be iterating over the transformed corpus_transformed multiple times, and the transformation is costly,
    # serialize the resulting corpus to disk first and continue using that.
    tfidf.save(path.join(mypath + "tutorial/original_tfidf.model"))


def lsi_transform(mypath):
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print('\n### lsi transform ###\n')
    if (os.path.exists(path.join(mypath + 'tutorial/original_dictionary.dict'))):
        dictionary = corpora.Dictionary.load(path.join(mypath + 'tutorial/original_dictionary.dict'))
        corpus_tfidf = corpora.MmCorpus(path.join(mypath + 'tutorial/corpus_tfidf.mm'))
        print("Used files generated from previous tutorial.")
    else:
        print("Please run first tutorial to generate data set.")

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=500)

    # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    # corpus_lsi = lsi[corpus_tfidf]

    lsi.print_topics(500)



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
loadDocument('D:/Dropbox/Dropbox_Uni/Europena/')
tfIdf_transform('D:/Dropbox/Dropbox_Uni/Europena/')
lsi_transform('D:/Dropbox/Dropbox_Uni/Europena/')
