import logging
import os
import sys
import unicodedata
from os import path
from pprint import pprint
from random import shuffle

import gensim
import nltk
import numpy
import pandas as pd
from bs4 import BeautifulSoup
from gensim import corpora
from gensim import utils
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from gensim.models.doc2vec import LabeledSentence
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nltk.tokenize import RegexpTokenizer
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from stop_words import get_stop_words

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


def print_tfidf(mypath):
    corpus_tfidf = corpora.MmCorpus(path.join(mypath + 'tutorial/original_corpus_tfidf.mm'))
    dictionary = corpora.Dictionary.load(path.join(mypath + 'tutorial/original_dictionary.dict'))

    # print words and their tfidf- values
    d = {}
    for doc in corpus_tfidf:
        for id, value in doc:
            word = dictionary.get(id)
            d[word] = value
    pprint(d)


def remove_single_words(texts):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info('removing words that appear only once')

    # remove words that appear only once
    frequency = nltk.defaultdict(int)

    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if (frequency[token] > 1)] for text in texts]
    return texts


def loadDocument(mypath):
    # METADATA_FILE = path.join(mypath + 'metadata_merged.csv')
    METADATA_FILE = path.join(mypath + 'metadata_translation_v2_groundtruth.csv')

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

    # pprint(documents) known_genre

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
                   '&sup3 tune(s) song(s) &lt;a href=&quot;http http wird übersetzt quot BNF unk bingham spart hebrew nucelli svec' \
                   'henebry kutchie ka hamills yanyor kavak hould fado clamper louys enzo angelillo yoshitomo kozlovskis '.split()
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

                # word must have more than one letter and must not be in the stop_word list
                if not (word in stop_words or (len(word) <= 1)):
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
    texts.append('\n')
    # end for document in documents

    # print(texts)
    texts = remove_single_words(texts)
    # texts = remove_unimportant_words(texts)

    return texts


def find_idf(mypath):
    corpus_tfidf = corpora.MmCorpus(path.join(mypath + 'tutorial/original_corpus_tfidf.mm'))
    dictionary = corpora.Dictionary.load(path.join(mypath + 'tutorial/original_dictionary.dict'))


def word2vec_clustering_words(mypath):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open(path.join(mypath, 'wordlist.txt'), 'r', encoding="UTF-8") as f:
        # sentences = [['first', 'sentence'], ['man', 'woman'], ['woman', 'king'], ['second', 'mozart'], ['man', 'hammer'],
        # ['king', 'lion'], ['beard', 'penis'], ['test', 'asdf'], ['rewq', 'tzujk']]


        texts = f.read().splitlines()
        lines = []
        for line in texts:
            lines.append(line.split(' '))

        # min_count...  min. appearances of word
        # size...       length of featurevecotr
        # workers...    cernels for determination
        # window...     # of words which are taken into account for determination
        model = Word2Vec(lines, min_count=3, size=200, workers=4, window=20)

        # vocab = list(model.vocab.keys())
        # print('vocab: ', vocab[:10])

        model.save(path.join(mypath, 'tutorial/model_persistanceTest.mm'))
        persistanceTest = gensim.models.Word2Vec.load(path.join(mypath, 'tutorial/model_persistanceTest.mm'))

        # model.build_vocab(sentences)  # can be a non-repeatable, 1-pass generator
        # model.train(sentences, min_count = 1)  # can be a non-repeatable, 1-pass generator
        # model = Word2Vec(sentences, min_count=1, size=5, workers=1)  # default value is 5

        # print('most_similar (postitive=[woman, king], negative=[man]: ', persistanceTest.most_similar(positive=['woman', 'king'], negative=['man'], topn=1))
        # print('similarity woman and man: ', persistanceTest.similarity('woman', 'man'))
        print('similarity wolfgang amadeus: ', persistanceTest.similarity('strauss', 'waltz'))
        print('\nSIMILARITY  bad:   ', persistanceTest.similarity('amadeus', 'folclor'))
        print('\nSIMILARITY good:   ', persistanceTest.similarity('amadeus', 'classic'))

        word_vectors = persistanceTest.syn0
        num_clusters = 6

        kmeans_clustering = KMeans(n_clusters=num_clusters, max_iter=500)
        idx = kmeans_clustering.fit_predict(word_vectors)

        # Create a Word / Index dictionary, mapping each vocabulary word to
        # a cluster number
        word_centroid_map = dict(zip(model.index2word, idx))

        # For the first 10 clusters
        for cluster in range(6):
            #
            # Print the cluster number
            print("\nCluster %d", cluster)
            #
            # Find all of the words for that cluster number, and print them out
            words = []
            for key in word_centroid_map:
                if (word_centroid_map[key] == cluster):
                    words.append(key)
            print(words)


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with open(path.join(mypath + '/tutorial', source), 'r', encoding="UTF-8") as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with open(path.join(mypath + '/tutorial', source), 'r', encoding="UTF-8") as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


def doc2vec_clustering_documents(mypath):
    # dictionary
    sources = {'test-classic.txt': 'TEST_CLASSIC', 'test-folklore.txt': 'TEST_FOLKLORE',
               'train-classic.txt': 'TRAIN_CLASSIC',
               'train-folklore.txt': 'TRAIN_FOLKLORE'}
    sentences = LabeledLineSentence(sources)

    # print(sentences.to_array()[1])
    # pprint(sentences.to_array())

    model = Doc2Vec(min_count=1, window=10, size=100, negative=5, workers=1)
    model.build_vocab(sentences.to_array())

    # train in different sequences
    for epoch in range(30):
        model.train(sentences.sentences_perm())

    # model.save(path.join(mypath+'/tutorial', 'dec2vec_model.d2v'))
    # model = Doc2Vec.load(path.join(mypath + '/tutorial', 'dec2vec_model.d2v'))

    train_arrays = numpy.zeros((24, 100))
    train_labels = numpy.zeros(24)

    for i in range(12):
        prefix_train_pos = 'TRAIN_CLASSIC_' + str(i)
        prefix_train_neg = 'TRAIN_FOLKLORE_' + str(i)
        train_arrays[i] = model.docvecs[prefix_train_pos]
        train_arrays[12 + i] = model.docvecs[prefix_train_neg]
        train_labels[i] = 1
        train_labels[12 + i] = 0

    test_arrays = numpy.zeros((24, 100))
    test_labels = numpy.zeros(24)

    for i in range(12):
        prefix_test_pos = 'TEST_CLASSIC_' + str(i)
        prefix_test_neg = 'TEST_FOLKLORE_' + str(i)
        test_arrays[i] = model.docvecs[prefix_test_pos]
        test_arrays[12 + i] = model.docvecs[prefix_test_neg]
        test_labels[i] = 1
        test_labels[12 + i] = 0

    classifier = LogisticRegression()
    classifier.fit(train_arrays, train_labels)

    print(classifier.score(test_arrays, test_labels))
    print(model.docvecs.most_similar('TEST_CLASSIC_0'))


'''
    #sentences = TaggedLineDocument(path.join(mypath, 'wordlist_test.txt'))
    doc1 = ["mozart violin composer opera ", "guitar dance irish drinking loud", "brestfeeding talking old man married interview"]

    docs = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for i, text in enumerate(doc1):
        words = text.lower().split(' ')
        tags = [i]
        docs.append(analyzedDocument(words, tags))
        print(words)



    model = Doc2Vec(docs, size = 100, window = 300, min_count = 1, workers = 4)
    #model.build_vocab(docs)
    #model.train(docs)

    print(model)
    pprint(model.most_similar('mozart'))


'''

#################### do Stuff
dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'mypath.txt')

with open(filename, 'r', encoding="UTF-8") as pathf:
    mypath = pathf.readline()

    doc2vec_clustering_documents(mypath)

    # word2vec_clustering_words(mypath)

'''
    with open(path.join(mypath, 'wordlist.txt'), 'r', encoding="UTF-8") as f:
        texts = f.read().splitlines()

        # do_kMeans(texts, 6)

        # print_tfidf()

        # text_ = Process_tfidf_lsi.remove_single_words(texts)
        # Process_tfidf_lsi.remove_unimportant_words(texts, mypath)


'''
