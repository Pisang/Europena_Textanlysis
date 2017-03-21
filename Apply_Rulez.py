import csv
import logging
import os
from os import path


def apply_rulez(mypath):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    csv.field_size_limit(500 * 1024 * 1024)

    with open(path.join(mypath, 'metadata_translation_v2.csv'), newline='', encoding="UTF-8") as translation:
        with open(path.join(mypath, 'metadata_genre_2017-01-03.csv'), 'w', newline='',
                  encoding="UTF-8") as metadata_genre:

            metadataReader = csv.reader(translation, delimiter=';')
            metadataList = list(metadataReader)

            first_row = True;

            for row in metadataList:
                genre = ''

                # skip titles
                if first_row:
                    first_row = False;
                    metadataWriter = csv.writer(metadata_genre, delimiter=';')
                    metadataWriter.writerow(row)
                    continue

                for column in row:
                    if ('mozart' in column.lower() or 'schubert, franz' in column.lower()):
                        genre = 'classical'

                    if ('testimony' in column.lower() and 'interview' in column.lower()):
                        genre = 'spoken word'
                    # A spoken life: [recorded autobiography between 1963 and 1994]
                    if (len(row[17].split(' ')) > 10):
                        if (row[17].split(' ')[4] == 'autobiography'):
                            genre = 'spoken word'
                    if (('free discussion' in row[18].lower() or 'interview' == row[18].lower())):
                        genre = 'spoken word'

                    if (row[16].lower() == 'instrumental folk music'):
                        genre = 'folklore'

                    if ('sound effect' in column.lower()):
                        genre = 'invironment'
                        # print(genre)

                row.append(genre)
                metadataWriter = csv.writer(metadata_genre, delimiter=';')
                metadataWriter.writerow(row)


dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'mypath.txt')

with open(filename, 'r', encoding="UTF-8") as pathf:
    mypath = pathf.readline()

    apply_rulez(mypath)
