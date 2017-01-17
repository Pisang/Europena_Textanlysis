import logging

import gensim
from os import path

mypath = 'D:/Dropbox/Dropbox_Uni/Europena/'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = gensim.models.Word2Vec.load(path.join(mypath + 'original_word2vec_model'))
print(model)
model.similarity('song', 'music')