import csv

import sys

import Translation_toolkit
import logging
import gensim
from os import path

# mypath = 'C:/Users/glask/Dropbox/Dropbox_Uni/Europena/'
import preprocessor

mypath = 'D:/Dropbox/Dropbox_Uni/Europena/'

##### Translate
myTranslator = Translation_toolkit
myTranslator.mypath = mypath
myTranslator.redo_faulty_translation()

##### Pre- processing
# pp = preprocessor
# texts = pp.__init__(pp, mypath)
#
# for row in texts:
#     print(row)
#
# ##### Train Word2Vec
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# model = gensim.models.Word2Vec(texts, size=100, window=10, min_count=5, workers=4)
# model.save(path.join(mypath, 'metadata_100x10x5'))







################################################################# do stuff:

# filterEnglishLines()

# translate_Metadata(0)
# translateToEnglish()

# print(translate_Google('schau ma mal ob das funktioniert'))

# filterColumn('description')
# filterColumn('format')
# filterColumn('subject')
# filterColumn('type')
# filterColumn('creator')
# filterColumn('contributor')
