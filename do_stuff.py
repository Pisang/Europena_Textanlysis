import logging
import os
from os import path

from gensim import corpora

import Process_tfidf_lsi

mypath = 'D:/Dropbox/Dropbox_Uni/Europena/'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
print('\n### lsi transform ###\n')
if (os.path.exists(path.join(mypath + 'tutorial/original_dictionary.dict'))):
    dictionary = corpora.Dictionary.load(path.join(mypath + 'tutorial/original_dictionary.dict'))
    corpus_tfidf = corpora.MmCorpus(path.join(mypath + 'tutorial/corpus_tfidf.mm'))
    print("Used files generated from previous tutorial.")
else:
    print("Please run first tutorial to generate data set.")

print(corpus_tfidf)
print(dictionary)

# vectorizer = TfidfVectorizer(min_df=1)
# X = vectorizer.fit_transform(corpus_tfidf)


Process_tfidf_lsi.do_dbscan(dictionary)

'''
model = gensim.models.Word2Vec.load(path.join(mypath + 'original_word2vec_model'))
for i in range(40):
    print(model.index2word[i])
#model.similarity('song', 'music')
'''
