import logging
import os
import sys
import unicodedata
from collections import defaultdict
from os import path

import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from gensim import corpora, models
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nltk.tokenize import RegexpTokenizer
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from stop_words import get_stop_words

wordlist = []

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
    # METADATA_FILE = path.join(mypath + 'metadata_merged.csv')
    METADATA_FILE = path.join(mypath + 'metadata_translation_v2.csv')

    metadata = pd.read_csv(METADATA_FILE, sep=";", encoding="utf-8")

    # convert nan-values to empty strings
    metadata = metadata.fillna("")

    # use the "id" field as index
    metadata = metadata.set_index("id")

    # concatinate to string:
    #   - (id)
    #   - contributor (not translated)
    #   - creator (only translated if country == france)
    #   - date
    #   - describtion - trans
    #   - spatial
    #   - subject - trans
    #   - type - trans
    #   - year
    documents = ((((((((metadata.contributor + " ").str.cat(metadata.creator) + " ").str.cat(
        metadata.date) + " ").str.cat(metadata.description) + " ").str.cat(metadata.spatial) + " ").str.cat(
        metadata.subject + " ").str.cat(metadata.type) + " ").str.cat(metadata.year) + " ").str.strip()).values

    # pprint(documents)

    stop_words = []
    stop_words.extend(get_stop_words('en'))
    stop_words.extend(get_stop_words('de'))
    stop_words.extend(get_stop_words('fr'))
    stop_words.extend(get_stop_words('it'))
    stop_words.extend(get_stop_words('pt'))
    stop_words.extend(get_stop_words('ro'))
    stop_words.extend(get_stop_words('spanish'))
    soup = BeautifulSoup('html')
    text = soup.get_text()
    stop_words.extend(text)

    my_stopwords = '00 000 01 02 03 03t14 03t18 04 04bravicr 04volamih 04vollusm 04volverr 05 06 07 08 09 09t12 1 10 100 ' \
                   '102 104 105 108 109 11 110 111 112 114 115 116 119 11942505816 11t11 12 120 12000 121 122 123 125 126 ' \
                   '128 13 130 132 135 138 139 14 140 144 14429 14e 15 150 154 155 16 160 161294 162 163 166 168 16864 17 ' \
                   '17085 173 176 17675 17796 17t12 18 180 183 184 189 18t12 19 190 191 192 193 194 195 198 19c 2 20 200 ' \
                   '20206 204 21 215 21588 216 21672 22 22205 22389 22435 22917 22936 22t20 23 230 232 23277 235 23986 24 ' \
                   '240 24808 25 250 252 257 26 264 27 271 27267 277 27t11 28 280 285 288 28t10 29 3 30 300 3000 30t11 31 ' \
                   '318 31t13 32 32a 33 330 3373 3m2 34 340 35 35471 359998 36 360 36279 36706 37 3700 370a 37271 378345 38 ' \
                   '38379 39 3o 4 40 400 4000 4060 40714 41 42 43 4300 4375309 439b 44 4445 45 450 46 463 4673 47 48 4814 ' \
                   '486 49 492 49403 50 500 50496 50s 51 51a 52 520 52002709 527 53 530 53473 54 543 55 5506 56 57 571 58 ' \
                   '58112 588 59 5r 60 600 61 610 62 620 63 64 65 6502 66 67 68 680 69 70 700 69th 70 700 71 72 72000 7278 ' \
                   '73 74 75 75019 76 77 78 79 79108 7rl 80 800 81 82 83 84 844 84968 84bajimbs 84bonmala 84cucbrem ' \
                   '84gracham 84lacpare 84lactrum 84merjour 84robamap 84routamg 84vitallm 84vitmoul 84vitolly 85 850222 ' \
                   '86 865 86b 87 87981 88 885219 89 89147 89437 90 900 91 91344 92 93 94 94629 95 96 97 98 99 9bis a a1 ' \
                   'a2 a7 aa aaa aaarr aacid aakjr aalge aasmund ab absolutely ac across actual actually ad06 add added ' \
                   'adding adds afterwards ago agree agreed agrees ah ai aim al already also always another anymore anyone ' \
                   'anything anywhere ap apr ar are arr atd ate au aui auk av ax ay b b1 b2 bab began begging begin begins ' \
                   'begun behind besides beyond big bigger biggest bn bnf bni bo br iz eg cl11 cl50 cl5 can us mr unknown ' \
                   'thirteen cd th are may unless otherwise tuesday january unlike dr almost although anymore anyone ' \
                   'anything anywhere appropriate appropriately &quot &untitled &apos &amp &quot &lt &gt &nbsp &iexcl &cent ' \
                   '&pound &curren &yen &brvbar &sect &uml &copy &ordf &laquo &not &shy &reg &macr &deg &plusmn &sup2 ' \
                   '&sup3 tune(s) song(s) &lt;a href=&quot;http http wird übersetzt quot BNF'.split()
    stop_words.extend(my_stopwords)
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

                # Grundformenreduktion
                try:
                    language = detect(word)
                except LangDetectException:
                    language = 'unknown'
                if language == 'en':
                    stemmer = nltk.stem.snowball.SnowballStemmer('english')
                    word = stemmer.stem(str(word))
                if language == 'de':
                    stemmer = nltk.stem.snowball.SnowballStemmer('german')
                    word = stemmer.stem(str(word))
                if language == 'it':
                    stemmer = nltk.stem.snowball.SnowballStemmer('italian')
                    word = stemmer.stem(str(word))
                if language == 'fr':
                    stemmer = nltk.stem.snowball.SnowballStemmer('french')
                    word = stemmer.stem(str(word))
                if language == 'nl':
                    stemmer = nltk.stem.snowball.SnowballStemmer('dutch')
                    word = stemmer.stem(str(word))

                # normalize, remove accents and umlaute
                word = unicodedata.normalize('NFKD', word).encode('ASCII', 'ignore')

                valid_words.append(word)

        texts.append(valid_words)
    # end for document in documents

    texts = remove_single_words(texts)

    # make the texts accassible for all methods
    wordlist = texts
    text_file = open(path.join(mypath + "wordlist.txt", "w"))
    for item in texts:
        text_file.write("%s\n" % item)
    text_file.close()

    dictionary = corpora.Dictionary(texts)
    dictionary.save(
        path.join(mypath + 'tutorial/original_dictionary.dict'))  # store the dictionary, for future reference

    print(dictionary)
    # Dictionary(71 unique tokens: ['comic', 'berlin', 'abraham', 'quot', 'romania']...)
    # pprint(SortedDict(dictionary.token2id))
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

    # doc_count = 0
    # for doc in corpus_tfidf:
    #    doc_count = doc_count + 1
    #    print(doc)
    # print(doc_count, ' documents are in TfIdf. ')

    # If you will be iterating over the transformed corpus_transformed multiple times, and the transformation is costly,
    # serialize the resulting corpus to disk first and continue using that.
    tfidf.save(path.join(mypath + "tutorial/original_tfidf.model"))

    # print words and their tfidf- values
    # corpus_tfidf = tfidf[corpus]
    # d = {}
    # for doc in corpus_tfidf:
    #     for id, value in doc:
    #         word = dictionary.get(id)
    #         d[word] = value
    # pprint(d)
    return tfidf

def lsi_transform(mypath):
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    print('\n### lsi transform ###\n')
    if (os.path.exists(path.join(mypath + 'tutorial/original_dictionary.dict'))):
        dictionary = corpora.Dictionary.load(path.join(mypath + 'tutorial/original_dictionary.dict'))
        corpus_tfidf = corpora.MmCorpus(path.join(mypath + 'tutorial/corpus_tfidf.mm'))
        print("Used files generated from previous tutorial.")
    else:
        print("Please run first tutorial to generate data set.")

    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)

    # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    # corpus_lsi = lsi[corpus_tfidf]

    lsi.print_topics(num_topics=5, num_words=15)



def count_languages(mypath):
    dictionary = corpora.Dictionary.load(path.join(mypath + 'tutorial/original_dictionary.dict'))
    languages = {'en': 0, 'fr': 0, 'de': 0, 'it': 0, 'pt': 0, 'ro': 0, 'spanish': 0, 'da': 0, 'cs': 0, 'et': 0, 'no': 0}

    for key, value in dictionary.items():
        try:
            language = detect(value)
            languages[language] = languages[language] + 1
            # if language == 'de': print(value)
        except LangDetectException:
            language = 'unknown'
        except KeyError:
            languages.update({language: 1})

    print(languages)


def do_dbscan(X):
    # X : array or sparse (CSR) matrix of shape (n_samples, n_features), or array of shape (n_samples, n_samples)
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(X)

    dbscan = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_samples_mask[dbscan.core_sample_indices_] = True
    labels = dbscan.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)

    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    ##############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
loadDocument('D:/Dropbox/Dropbox_Uni/Europena/')
tfidf = tfIdf_transform('D:/Dropbox/Dropbox_Uni/Europena/')
lsi_transform('D:/Dropbox/Dropbox_Uni/Europena/')

do_dbscan(wordlist)

#count_languages('D:/Dropbox/Dropbox_Uni/Europena/')
