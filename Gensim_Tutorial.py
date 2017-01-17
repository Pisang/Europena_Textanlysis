from os import path

from gensim import corpora


def buildCorpus(mypath):
    # logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    documents = ["Human machine interface for lab abc computer applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"]

    ### First, lets tokenize the documents (remove stop words and words that only appear once)
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]

    # [['human', 'machine', 'interface', 'lab', 'abc', 'computer', 'applications'],
    #  ['survey', 'user', 'opinion', 'computer', 'system', 'response', 'time'], [...]]

    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]

    # [['human', 'interface', 'computer'],
    # ['survey', 'user', 'computer', 'system', 'response', 'time'], [...]]

    ### To convert documents to vectors we use BAG-OF-WORDS:
    ### -> each document is represented as one vector where each vector element represents a question-answer pair in the
    ###    style of "How many times does the word X appear in the document?"   -   > Once <
    ###    e.g. word #1 - 4 times
    ###         word #2 - 9 times ...
    ###
    ### The vectors are stored as integers - as 'dictionary' -> each word gets an unique ID.
    ### Here after removing stop words, ... 12 distinct words remain
    ### -> each document is represented by 12 numbers (12 D vector)

    dictionary = corpora.Dictionary(texts)
    dictionary.save(path.join(mypath + 'tutorial/deerwester.dict'))  # store the dictionary, for future reference
    print(dictionary)
    # Dictionary(12 unique tokens: ['user', 'system', 'eps', 'survey', 'interface']...)
    print(dictionary.token2id)
    # {'user': 3, 'system': 4, 'eps': 8, ...}

    ### try with new document
    new_doc = "Human computer interaction"
    new_vec = dictionary.doc2bow(new_doc.lower().split())
    print(new_vec)

    # The function doc2bow() simply counts the number of occurrences of each distinct word, converts the word to its
    # integer word id and returns the result as a sparse vector. The sparse vector [(0, 1), (1, 1)] therefore reads:
    # in the document “Human computer interaction”, the words computer (id 0) and human (id 1) appear once; the other
    # ten dictionary words appear (implicitly) zero times

    corpus = [dictionary.doc2bow(text) for text in texts]
    corpora.MmCorpus.serialize(path.join(mypath + 'tutorial/deerwester.mm'), corpus)
    print(corpus)


buildCorpus('D:/Dropbox/Dropbox_Uni/Europena/')