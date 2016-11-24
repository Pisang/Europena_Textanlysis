from os import path
import pandas as pd
from nltk.tokenize import RegexpTokenizer

mypath = ''
METADATA_FILE = ''
metadata = ''

def __init__(self, mypath):
    self.mypath = mypath
    METADATA_FILE = path.join(mypath + 'metadata.csv')
    # read csv-data (separated by semicolons)
    metadata = pd.read_csv(METADATA_FILE, sep=";", encoding="utf-8")

# convert nan-values to empty strings
metadata = metadata.fillna("")



