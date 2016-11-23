import csv
import sys

from os import path

# mypath = 'C:/Users/glask/Dropbox/Dropbox_Uni/Europena/'
# mypath = 'D:/Dropbox/Dropbox_Uni/Europena/'

mypath = ''


def __init__(self, mypath):
    self.mypath = mypath


def serchGenresInMetadata():
    '''
    In this Method for each line of the Methadata CSV genres are serched.
    '''
    genreFile = open(path.join(mypath, 'genres.txt'), 'r')
    genres = genreFile.read()

    csv.field_size_limit(500 * 1024 * 1024)
    with open(path.join(mypath, 'metadata.csv'), newline='', encoding="UTF-8") as csvfile:
        metadataReader = csv.reader(csvfile, delimiter=';')
        space = " "

        textfile_found_genres = open('Found_Genres.txt', 'w')
        rowcount = 0
        found_genres = ''
        for row in metadataReader:

            # output percentage
            rowcount += 1
            percent = round(100 / 9571 * rowcount, 2)
            sys.stdout.write('\r')
            sys.stdout.write(str(percent) + '%')
            sys.stdout.flush()
            ########################################

            line = space.join(row)
            for genre in genres:
                if genre in line:
                    # found_genres = ''.join(line)
                    textfile_found_genres.write(repr(line))

        # print('\n' + str(i) + ' times one of the genres have been found in the metadata.')

        textfile_found_genres.close()


################################################################# do stuff:

# filterEnglishLines()

translate_Metadata(0)
# translateToEnglish()

# print(translate_Google('schau ma mal ob das funktioniert'))

# filterColumn('description')
# filterColumn('format')
# filterColumn('subject')
# filterColumn('type')
# filterColumn('creator')
# filterColumn('contributor')
