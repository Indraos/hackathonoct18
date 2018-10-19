import csv
import nltk
import json
import statistics, string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from os import listdir
from os.path import isfile, join
import sys
from dateutil.parser import parse
# import WordEmbedding

PATH_REUTER = r'..\Reuters_US'
w2v_path = ''

# w2v_model = WordEmbedding.EmbeddingModel(300, w2v_path, True, False)
csv.field_size_limit(100000000)


all_paths = [join(PATH_REUTER, f) for f in listdir(PATH_REUTER) if isfile(join(PATH_REUTER, f))]
# for path in all_paths:
#     if '2017' in path:
#         print(path)

print(all_paths[0])
time_stamp_to_sent = {}
for file in all_paths[:2]:
    with open(file, 'r', encoding="utf8") as csvfile:
        a = csv.reader(csvfile, delimiter=';')
        for row in a:
            if row[0] is not "":
                time_stamp = parse(row[0], fuzzy_with_tokens=True)
                headers = row[1].replace('-', ' ')
                time_stamp_to_sent[time_stamp[0]] = nltk.word_tokenize(headers)

with open('./files/headers_as_list', 'w') as file:
    json.dump(time_stamp_to_sent, file)

