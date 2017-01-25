import csv
import sys
import time
from os import path
from traceback import print_exception

from bs4 import BeautifulSoup
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from microsofttranslator import Translator
from selenium import webdriver

mypath = ''


def __init__(self, mypath):
    self.mypath = mypath


def filterEnglishLines():
    '''
        In this Method lines with english language are filtered out of the metadata.csv. The remaining rows are
        written into the metadata_to_translate.csv for further translation. This is done to minimize the words
        that are translated because bing limits the translation API to 2 Mio letters / month.
    '''
    with open(path.join(mypath, 'metadata.csv'), newline='', encoding="UTF-8") as metadata_csv:
        with open(path.join(mypath, 'metadata_to_translate.csv'), 'w', newline='',
                  encoding="UTF-8") as metadta_to_translate_csv:
            csv.field_size_limit(500 * 1024 * 1024)
            metadataReader = csv.reader(metadata_csv, delimiter=';')
            metadataList = list(metadataReader)

            count_row = 0
            count_english = 0
            for row in metadataList:
                if not (row[10] == 'English' or row[10] == 'en' or row[5] == 'The British Library' or row[
                    2] == 'ireland'):
                    metadataWriter = csv.writer(metadta_to_translate_csv, delimiter=';')
                    metadataWriter.writerow(row)
        metadta_to_translate_csv.close()
    metadata_csv.close()


def translateToEnglish():
    '''
    just translate the fiesls:
        - describtion
        - format
        - subject
        - type
        - if country == france : creator
    and only translate the cells if there isn't already english language in it.
    '''
    translateText = open(path.join(mypath, '/testFile.txt'), 'w', encoding="UTF-8")
    with open(path.join(mypath, 'metadata_to_translate.csv'), newline='',
              encoding="UTF-8") as metadata_csv:
        csv.field_size_limit(500 * 1024 * 1024)
        metadataReader = csv.reader(metadata_csv, delimiter=';')
        metadataList = list(metadataReader)

        count_exceptions = 0

        for row in metadataList:
            # if country == france, translate the creator
            try:
                language = detect(row[4])
            except LangDetectException:
                count_exceptions += 1;

            if (row[2] and row[2] == 'france' and language != 'en'):
                # translate
                translateText.write(row[2])

            # translate describtion
            try:
                language = detect(row[7])
            except LangDetectException:
                count_exceptions += 1;
            if (row[7] and language != 'en'):
                # translate
                translateText.write(row[7])

            # translate format
            try:
                language = detect(row[8])
            except LangDetectException:
                count_exceptions += 1;
            if (row[8] and language != 'en'):
                # translate
                translateText.write(row[8])

            # translate subject
            try:
                language = detect(row[8])
            except LangDetectException:
                count_exceptions += 1;
            if (row[16] and language != 'en'):
                # translate
                translateText.write(row[16])

            # translate type
            try:
                language = detect(row[8])
            except LangDetectException:
                count_exceptions += 1;
            if (row[18] and language != 'en'):
                # translate
                translateText.write(row[18])
    translateText.close()


def translate_Metadata(skip_lines):
    '''
    Translate the data. In the first run the following is translated:
    - if dataProvider [5] == 'CNRS-MMSH' or 'Bibliothèque Nationale de France' or 'National Library of France' :
      --> translate creator [4] and contributor [1]
    - describtion [7],
    - format [8],
    - subject [16] and
    - type [18]

    but afterwards also
    - title [17]
    - spacial [15]
    - some greek describtions mixed with english which have not been translation because languages are
      not translated if they are mixed
    should be translated. therefore a mode is introduced:

    mode == 1  --> first run
    mode == 2  --> second run

    :param skip_lines:
    :return:
    '''

    mode = 2

    # with open(path.join(mypath, 'TEST_translation.csv'), 'w', newline='', encoding="UTF-8") as writer:
        # with open(path.join(mypath, 'TEST_translate1.csv'), 'r', newline='', encoding="UTF-8") as metadata:

    with open(path.join(mypath, 'metadata_translation_v2.csv'), 'w', newline='', encoding="UTF-8") as writer:
        with open(path.join(mypath, 'metadata_merged.csv'), 'r', newline='', encoding="UTF-8") as metadata:
            csv.field_size_limit(500 * 1024 * 1024)
            metadataReader = csv.reader(metadata, delimiter=';')
            metadataList = list(metadataReader)

            skipfirstline = True
            metadataWriter = csv.writer(writer, delimiter=';')

            # skip lines for running working step by step
            skipped_lines = 0
            rowcount = 0
            for row in metadataList:
                if skip_lines > skipped_lines:
                    skipped_lines += 1
                    continue

                # first line are headers -> skip them
                if (skipfirstline and skip_lines == 0):
                    metadataWriter.writerow(row)
                    skipfirstline = False
                    continue

                # output percentage
                rowcount += 1
                percent = round(100 / len(metadataList) * rowcount, 2)
                sys.stdout.write('\r')
                sys.stdout.write('Translation in progress: ' + str(percent) + '%\n')
                sys.stdout.flush()
                ########################################

                translated_row = []
                for i in range(len(row)):
                    line = row[i]
                    translated_line = line

                    # first run translate several fields
                    if mode == 1:
                        # if dataProvider [5] == 'CNRS-MMSH' or 'Bibliothèque Nationale de France' or 'National Library of France' :
                        # translate creator [4] and contributor [1]
                        if (i == 1 and (row[5] == 'CNRS-MMSH' or row[5] == 'Bibliothèque Nationale de France' or row[
                            5] == 'National Library of France')):
                            if (line != ''):
                                translated_line = translate_Google(line)
                        if (i == 4 and (row[5] == 'CNRS-MMSH' or row[5] == 'Bibliothèque Nationale de France' or row[
                            5] == 'National Library of France')):
                            if (line != ''):
                                translated_line = translate_Google(line)
                        # translate describtion [7], format [8], subject [16] and type [18]
                        if (i == 7 or i == 8 or i == 16 or i == 18):
                            if (line != ''):
                                translated_line = translate_Google(line)

                    # for second run
                    # - title [17] and
                    # - spacial [15] is added
                    # - some greek describtions mixed with english which have not been translation because languages are
                    #   not translated if they are mixed
                    if mode == 2:
                        translate_this_part = ''
                        try:
                            language = detect(line)
                        except LangDetectException:
                            print('nothing to detect')
                        if ((i == 15 or i == 17) and language != 'en'):
                            if (line != ''):
                                translated_line = translate_Google(line)

                        if (i == 7 and row[3] == 'greece'):
                            words = line.split(' ')
                            greekcount = 0
                            for word_index in range(len(words)):
                                testword = words[word_index]
                                try:
                                    language = detect(testword)
                                except LangDetectException:
                                    print('nothing to detect')
                                if(language == 'el'):
                                    greekcount = greekcount+1

                            if greekcount > 2:
                                print('greeks found!')
                                translated_line = translate_Google_greek(line)

                            print('line: ' + translated_line)

                    translated_row.append(translated_line)
                metadataWriter.writerow(translated_row)


def translate_Bing(text):
    '''
    This method translates a text string to english with the Microsoft Bing_translator.
    Note that this translator only translates 2.000.000 Letters / Month for free using
    a Microsoft account and the registered client with secret.

    :param text: String that is to be translated
    :return: english translation of the text
    '''
    client_id = 'EuropenaTranslator'
    client_secret = 'iAcywBlP37WpSs/qxNVdlCOjVXYSti+L9YTUvBx4ets='

    translator = Translator(client_id, client_secret)
    text4translation = translator.translate(text, 'en')

    return text4translation


def translate_Google(line):
    translation = line
    try:
        browser = webdriver.Firefox()
        # browser = webdriver.PhantomJS('C:/Program Files/phantomjs-2.1.1-windows/bin/phantomjs.exe')

        browser.get('http://translate.google.com/#auto/en/' + line)
        time.sleep(2)

        html_content = browser.page_source
        soup = BeautifulSoup(html_content, "html.parser")
        # soup = BeautifulSoup(html_content, "html5lib")

        result_span = soup.find_all('span', class_='short_text')
        # if text is longer, the result varies
        if (len(result_span) == 0):
            result_span = soup.find_all(id='result_box')
        translation = result_span[0].text
        if (len(result_span) == 0):
            print('didnt find the right box!')
    except Exception as e:
        print_exception(*sys.exc_info())
    finally:
        browser.quit()
        return translation

def translate_Google_greek(line):
    translation = line
    try:
        browser = webdriver.Firefox()
        # browser = webdriver.PhantomJS('C:/Program Files/phantomjs-2.1.1-windows/bin/phantomjs.exe')

        browser.get('http://translate.google.com/#el/en/' + line)
        time.sleep(5)

        html_content = browser.page_source
        soup = BeautifulSoup(html_content, "html.parser")
        # soup = BeautifulSoup(html_content, "html5lib")

        result_span = soup.find_all('span', class_='short_text')
        # if text is longer, the result varies
        if (len(result_span) == 0):
            result_span = soup.find_all(id='result_box')
        translation = result_span[0].text
        if (len(result_span) == 0):
            print('didnt find the right box!')
    except Exception as e:
        print_exception(*sys.exc_info())
    finally:
        browser.quit()
        return translation

def filterColumn(column_name):
    '''
    This method prints out just one particular column of the metadata into a utf-8 .txt file
    to copy- paste it into google translate.

    :param column_name: name of the metadata- column which should be seperated into a utf-8 file
    '''
    translateText = open(path.join(mypath, '\original_' + column_name + '.txt'), 'w',
                         encoding="UTF-8")
    with open(path.join(mypath, 'metadata_to_translate.csv'), newline='',
              encoding="UTF-8") as metadata_csv:
        csv.field_size_limit(500 * 1024 * 1024)
        metadataReader = csv.reader(metadata_csv, delimiter=';')
        metadataList = list(metadataReader)

        for row in metadataList:
            if (column_name == 'id'):
                translateText.writelines(row[0] + '\n')
            if (column_name == 'contributor'):
                translateText.writelines(row[1] + '\n')
            if (column_name == 'country'):
                translateText.writelines(row[2] + '\n')
            if (column_name == 'created'):
                translateText.writelines(row[3] + '\n')
            if (column_name == 'creator'):
                translateText.writelines(row[4] + '\n')
            if (column_name == 'dataProvider'):
                translateText.writelines(row[5] + '\n')
            if (column_name == 'date'):
                translateText.writelines(row[6] + '\n')
            if (column_name == 'description'):
                translateText.writelines(row[7] + '\n')
            if (column_name == 'format'):
                translateText.writelines(row[8] + '\n')
            if (column_name == 'identifier'):
                translateText.writelines(row[9] + '\n')
            if (column_name == 'language'):
                translateText.writelines(row[10] + '\n')
            if (column_name == 'medium'):
                translateText.writelines(row[11] + '\n')
            if (column_name == 'provider'):
                translateText.writelines(row[12] + '\n')
            if (column_name == 'publisher'):
                translateText.writelines(row[13] + '\n')
            if (column_name == 'relation'):
                translateText.writelines(row[14] + '\n')
            if (column_name == 'spatial'):
                translateText.writelines(row[15] + '\n')
            if (column_name == 'subject'):
                translateText.writelines(row[16] + '\n')
            if (column_name == 'title'):
                translateText.writelines(row[17] + '\n')
            if (column_name == 'type'):
                translateText.writelines(row[18] + '\n')
            if (column_name == 'year'):
                translateText.writelines(row[19] + '\n')
    translateText.close()


def redo_faulty_translation():
    '''
    At the first iteration about 1.964 cells were translatet to "Wird übersetzt'. Probably because there was not enough
    time for the translation to load. Therefore those lines are filtered out and translated a second time, where I extend
    the time the browser waits for the translation to be complete manually.

    :return:
    '''
    with open(path.join(mypath, 'metadata_translation_renewed.csv'), 'w', newline='', encoding="UTF-8") as writer:
        with open(path.join(mypath, 'metadata_translation_0-18042_faulty.csv'), 'r', newline='', encoding="UTF-8") as metadata_faulty:
            with open(path.join(mypath, 'metadata_to_translate.csv'), 'r', newline='', encoding="UTF-8") as metadata_original:

                csv.field_size_limit(500 * 1024 * 1024)
                metadataReader_faulty = csv.reader(metadata_faulty, delimiter=';')
                metadataList_faulty = list(metadataReader_faulty)

                metadataReader_original = csv.reader(metadata_original, delimiter=';')
                metadataList_original = list(metadataReader_original)

                metadataWriter = csv.writer(writer, delimiter=';')

                rowcount = 0
                for row in metadataList_faulty:

                    ####################################### progress in percent
                    rowcount += 1
                    percent = round(100 / len(metadataList_faulty) * rowcount, 0)
                    sys.stdout.write('\r')
                    sys.stdout.write('Translation in progress: ' + str(percent) + '%')
                    sys.stdout.flush()
                    ############################################################

                    translated_row = []
                    for cell_f in range(len(row)):
                        line = row[cell_f]
                        translated_line = line

                        if (line == 'Wird übersetzt...'):
                            line_id_faulty = row[0]

                            for row_original in metadataList_original:

                                line_id_original = row_original[0]
                                if (line_id_original == line_id_faulty):
                                    translated_line = translate_Google(row_original[cell_f])
                        translated_row.append(translated_line)
                    metadataWriter.writerow(translated_row)


def merge_metadata():
    with open(path.join(mypath, 'metadata_merged.csv'), 'w', newline='', encoding="UTF-8") as writer:
        with open(path.join(mypath, 'metadata_translation_renewed.csv'), 'r', newline='', encoding="UTF-8") as metadata_renewed:
            with open(path.join(mypath, 'metadata.csv'), 'r', newline='', encoding="UTF-8") as metadata:

                csv.field_size_limit(500 * 1024 * 1024)
                metadataReader_renewed = csv.reader(metadata_renewed, delimiter=';')
                metadata_renewed = list(metadataReader_renewed)

                metadataReader = csv.reader(metadata, delimiter=';')
                metadataList = list(metadataReader)

                metadataWriter = csv.writer(writer, delimiter=';')

                rowcount = 0

                for row_original in metadataList:
                    row = row_original

                    ####################################### progress in percent
                    rowcount += 1
                    percent = round(100 / len(metadataList) * rowcount, 0)
                    sys.stdout.write('\r')
                    sys.stdout.write('Translation in progress: ' + str(percent) + '%')
                    sys.stdout.flush()
                    ############################################################

                    count = 0
                    for row_ren in metadata_renewed:
                        if (row_original[0] == row_ren[0]):
                            row = row_ren
                    metadataWriter.writerow(row)


mypath = 'C:/Users/glask/Dropbox/Dropbox_Uni/Europena/'
# mypath = 'D:/Dropbox/Dropbox_Uni/Europena/'

translate_Metadata(0)