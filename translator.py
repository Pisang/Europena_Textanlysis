import Translation_toolkit

# mypath = 'C:/Users/glask/Dropbox/Dropbox_Uni/Europena/'
import preprocessor

mypath = 'D:/Dropbox/Dropbox_Uni/Europena/'

##### Translate
#myTranslator = Translation_toolkit
#myTranslator.mypath = mypath
#myTranslator.translate_Metadata(2884)

##### Preprocessing
pp = preprocessor
pp.__init__(pp, mypath)











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
