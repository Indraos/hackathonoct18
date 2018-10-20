import os, pickle
import numpy as np
from collections import Counter
import pandas as pd
import Models
import random
import re
from sklearn import metrics
import WordEmbedding
import nltk, keras
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt


HEADER_LENGTH = 16

SENT_LENGTH = 25
TOL = 0.005
tokenizer = RegexpTokenizer(r'\w+')

path_to_embeddings = r'./dense_headlines/'
path_to_df = r'./data/'

def train():
    # w2v_path = r'C:\Users\fkarl\PycharmProjects\Image2SequenceFiles\w2vModel\wiki.en.vec'
    # w2v_path = r'C:\Users\fkarl\PycharmProjects\Image2SequenceFiles\w2vModel\wiki-news-300d-1M-subword.vec'
    w2v_path = r'C:\Users\fkarl\PycharmProjects\Image2SequenceFiles\w2vModel\en.wiki.bpe.op200000.d300.w2v.bin'
    w2v_model = WordEmbedding.EmbeddingModel(300, w2v_path, True, True)


    YEAR_START = 2010
    YEAR_END = 2017


    model = Models.get_stock_pred_model(None, HEADER_LENGTH, SENT_LENGTH)
    model.summary()
    epochs = 10
    for i in range(epochs):
        print('EPOCH: ' + str(i) + ' of ' + str(epochs))
        for year in range(YEAR_START,YEAR_END):
            print('Year: ' + str(year))
            df_year_path = path_to_df + str(year)

            df_paths_per_month = [os.path.join(df_year_path, f) for f in os.listdir(df_year_path) if os.path.isfile(os.path.join(df_year_path, f))]
            print(df_paths_per_month)


            rand_int = random.randint(0,len(df_paths_per_month)-1)
            val_df = df_paths_per_month[rand_int]
            df_paths_per_month.remove(val_df)

            with open(val_df, 'rb') as f:
                data_frame = pickle.loads(f.read())

            data_frame = refine_dataframe(data_frame)
            # val_data_frame = val_data_frame[val_data_frame['text'].filter(regex=('tech'), axis=0)]

            Y_val = data_frame['BA'].values
            Y_val = get_targets(Y_val)
            print('ValidationSet',Counter(list(Y_val)))

            headers = data_frame['abstract'].map(str).map(str.lower).map(nltk.word_tokenize).values
            embeddings = [np.array(w2v_model.get_embeddings(sent)) for sent in headers]
            padded_headders = keras.preprocessing.sequence.pad_sequences(embeddings, maxlen=HEADER_LENGTH, dtype='float32', padding='post', truncating='post', value=0.0)
            X_headders_val = np.array(padded_headders)

            X_sents_val = get_refined_news(data_frame, w2v_model)



            assert(len(X_headders_val) == len(Y_val))

            for data_frame_path in df_paths_per_month:

                with open(data_frame_path, 'rb') as f:
                    data_frame = pickle.loads(f.read())

                data_frame = refine_dataframe(data_frame)


                Y = data_frame['BA'].values
                Y = get_targets(Y)
                print('TrainingSet',Counter(list(Y)))

                # with open(embedding_path, 'rb') as f:
                #     embedded_words = pickle.loads(f.read())

                headers = data_frame['abstract'].map(str).map(str.lower).map(nltk.word_tokenize).values
                embeddings = [np.array(w2v_model.get_embeddings(sent)) for sent in headers]
                padded_headders = keras.preprocessing.sequence.pad_sequences(embeddings, maxlen=HEADER_LENGTH, dtype='float32', padding='post', truncating='post', value=0.0)
                X_headders = np.array(padded_headders)

                X_sents = get_refined_news(data_frame, w2v_model)

                assert(len(Y) == len(X_headders))
                assert(len(headers) == len(X_headders))


                print('*** Training *** on ' + data_frame_path + ' with ' + str(len(Y)) + ' samples')

                model.fit([X_headders, X_sents], Y, batch_size=16, validation_split=0.05, verbose=2, epochs=3)
                model.save('stock_pred_model.h5')

                print('*** Validation ***')
                predictions = model.predict([X_headders_val, X_sents_val], batch_size=64)
                rounded_predicitons = [np.round(elem[0]) for elem in predictions]
                # conservative_results = [1 if elem >= TOL else 0 for elem in predictions]
                print('TestResults', Counter(rounded_predicitons))
                print(metrics.precision_recall_fscore_support(Y_val, rounded_predicitons))

                fpr, tpr, threshold = metrics.roc_curve(Y_val, [elem[0] for elem in predictions])
                roc_auc = metrics.auc(fpr, tpr)

                plt.title('Receiver Operating Characteristic')
                plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
                plt.legend(loc='lower right')
                plt.plot([0, 1], [0, 1], 'r--')
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.ylabel('True Positive Rate')
                plt.xlabel('False Positive Rate')
                plt.show()


def get_refined_news(data_frame,w2v_model):
    news_articles = data_frame['text'].values
    embeddings = []
    try:
        for article in news_articles:
            article = str(article)
            if '-' in article:
                article = article[article.index("-") + 1:]
            article = tokenizer.tokenize(str(article).lower())
            news = [word for word in article[:SENT_LENGTH + 10] if len(word) > 3]
            embeddings.append(np.array(w2v_model.get_embeddings(news)))
        padded_sents = keras.preprocessing.sequence.pad_sequences(embeddings, maxlen=SENT_LENGTH, dtype='float32', padding='post', truncating='post', value=0.0)
    except TypeError as error:
        print(article)
    return np.array(padded_sents)


def refine_dataframe(data_frame):
    data_frame.dropna()
    # data_frame = data_frame[~data_frame['abstract'].isin(['UPDATE'])]
    data_frame['text'] = data_frame['text'].str.lower()
    data_frame = data_frame[data_frame['text'].str.contains('airbus')| data_frame['text'].str.contains('lockheed martin')| data_frame['text'].str.contains('Embraer')| data_frame['text'].str.contains('plain')| data_frame['text'].str.contains('air')]
    # data_frame = data_frame[
    #     data_frame['text'].str.contains('apple') | data_frame['text'].str.contains('iphone') | data_frame['text'].str.contains('itunes') | data_frame['text'].str.contains('icloud') | data_frame['text'].str.contains('ipad') | data_frame[
    #         'text'].str.contains('boeing') | data_frame['text'].str.contains('ba.n') | data_frame[
    #         'text'].str.contains('airbus') | data_frame['text'].str.contains('a380')| data_frame['text'].str.contains('a380')| data_frame['text'].str.contains('plainmaker')]
    # data_frame = data_frame[
    #     data_frame['text'].str.contains('ba.n') | data_frame['text'].str.contains('boeing') | data_frame['text'].str.contains(
    #         'airbus') | data_frame['text'].str.contains('a380') | data_frame['text'].str.contains('plainmaker')]
    return data_frame


def get_targets(Y_val):
    return [1 if elem >= TOL or elem <= -TOL else 0 for elem in Y_val]
    # return [1 if elem >= 0.0 else 0 for elem in Y_val]


# create_embedded_headdings()
train()


