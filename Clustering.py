import logging
import os
from os import path
from pprint import pprint

from gensim import corpora
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import Process_tfidf_lsi

# mypath = 'D:/Dropbox/Dropbox_Uni/Europena/'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def do_dbscan(X):
    db = DBSCAN(eps=0.1, min_samples=100).fit(X)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)


def do_kMeans(texts, k_cluster):
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(texts)

    model = KMeans(n_clusters=k_cluster, init='k-means++', max_iter=500, n_init=10)
    model.fit(X)

    print("Top terms per cluster:")
    # cluster_centers_ : array, [n_clusters, n_features]
    terms = vectorizer.get_feature_names()
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    for j in range(k_cluster):
        print("\nCluster %d:" % j)
        for i in order_centroids[j, :10]:
            print('%s' % terms[i])

    prediction = model.predict(X)
    labels = model.labels_
    # Number of clusters in labels, ignoring noise if present.
    # print(prediction)

    for document in range(len(texts)):
        print('Document ', "%02d" % (document + 1), ' : ', prediction[document], ' - ', texts[document])


def print_tfidf():
    corpus_tfidf = corpora.MmCorpus(path.join(mypath + 'tutorial/original_corpus_tfidf.mm'))
    dictionary = corpora.Dictionary.load(path.join(mypath + 'tutorial/original_dictionary.dict'))

    # print words and their tfidf- values
    d = {}
    for doc in corpus_tfidf:
        for id, value in doc:
            word = dictionary.get(id)
            d[word] = value
    pprint(d)


dir = os.path.dirname(__file__)
print(dir)
filename = os.path.join(dir, 'mypath.txt')

with open(filename, 'r', encoding="UTF-8") as pathf:
    mypath = pathf.readline()
    with open(path.join(mypath, 'wordlist.txt'), 'r', encoding="UTF-8") as f:
        texts = f.read().splitlines()

        # do_kMeans(texts, 6)

        # print_tfidf()

        # text_ = Process_tfidf_lsi.remove_single_words(texts)
        Process_tfidf_lsi.remove_unimportant_words(texts, mypath)
