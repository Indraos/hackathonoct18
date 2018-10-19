import csv
import nltk
import json
import statistics, string
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from os import listdir
from os.path import isfile, join
import sys, pickle
import pandas
from dateutil.parser import parse
import WordEmbedding
import keras

PATH_REUTER = r'..\Reuters_US'
w2v_path = r'C:\Users\fkarl\PycharmProjects\Image2SequenceFiles\w2vModel\en.wiki.bpe.op200000.d300.w2v.bin'

w2v_model = WordEmbedding.EmbeddingModel(300, w2v_path, True, True)
csv.field_size_limit(100000000)


all_paths = [join(PATH_REUTER, f) for f in listdir(PATH_REUTER) if isfile(join(PATH_REUTER, f))]
paths_from_2017 = [path for path in all_paths if '2017' in path ]

time_stamps=[]
sentences = []
for file in paths_from_2017[:2]:
    with open(file, 'r', encoding="utf8") as csvfile:
        a = csv.reader(csvfile, delimiter=';')
        for row in a:
            if row[0] is not "":
                time_stamps.append(parse(row[0], fuzzy_with_tokens=True))
                sentences.append(w2v_model.get_embeddings(row[1].replace('-', ' ')))


paddded_sentence = keras.preprocessing.sequence.pad_sequences(sentences, maxlen=20, dtype='int32', padding='post', truncating='post', value=0.0)

time_stamp_sent_dict = {time_stamp:sent for time_stamp, sent in zip(time_stamps,paddded_sentence)}

data_frame = pandas.DataFrame(time_stamp_sent_dict)

with open('./files/headers_as_list', 'w') as file:
    pickle.dump(data_frame, file)

