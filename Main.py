import os, pickle
import numpy as np
from collections import Counter
import pandas as pd
import Models
import random
from sklearn import metrics
import WordEmbedding
import nltk, keras



SENTENCE_SHAPE = (20,300)



path_to_embeddings = r'./dense_headlines/'
path_to_df = r'./data/'

# def calculate_embedding():
#     sent_lists = list(data['abstract'].map(str.lower).map(nltk.word_tokenize))
#     embeddings = [np.array(w2v_model.get_embeddings(sent)) for sent in sent_lists]
#     padded_sents = keras.preprocessing.sequence.pad_sequences(embeddings,maxlen=20, dtype='float32', padding='post', truncating='post', value=0.0)
#     print(len(padded_sents))
#     print(np.array(padded_sents).shape)


def create_embedded_headdings():
    w2v_path = r'C:\Users\fkarl\PycharmProjects\Image2SequenceFiles\w2vModel\wiki-news-300d-1M-subword.vec'
    # w2v_path = r'C:\Users\fkarl\PycharmProjects\Image2SequenceFiles\w2vModel\en.wiki.bpe.op200000.d300.w2v.bin'
    w2v_model = WordEmbedding.EmbeddingModel(300, w2v_path, True, False)

    for folder in os.listdir(path_to_df):
        current_year = folder
        if int(current_year) > 2012:
            year_folder_path = os.path.join(path_to_df, folder)
            for moth_file_path in os.listdir(year_folder_path):
                current_month = moth_file_path
                complete_path = os.path.join(year_folder_path, moth_file_path)
                print(complete_path)
                with open(complete_path, 'rb') as f:
                    df_for_month = pickle.loads(f.read())

                sent_lists = list(df_for_month['abstract'].map(str).map(str.lower).map(nltk.word_tokenize))
                embeddings = [np.array(w2v_model.get_embeddings(sent)) for sent in sent_lists]
                padded_sents = keras.preprocessing.sequence.pad_sequences(embeddings, maxlen=20, dtype='float32', padding='post', truncating='post', value=0.0)
                print(len(padded_sents))
                print(np.array(padded_sents).shape)

                save_folder = os.path.join(path_to_embeddings, folder)
                os.makedirs(save_folder, exist_ok=True)
                save_name = os.path.join(save_folder, current_month)
                with open(save_name, 'wb') as file:
                    pickle.dump(padded_sents, file)



def train():
    YEAR_START = 2009
    YEAR_END = 2013


    model = Models.get_stock_pred_model(None, SENTENCE_SHAPE)
    model.summary()
    epochs = 10
    for i in range(epochs):
        print('EPOCH: ' + str(i) + ' of ' + str(epochs))
        for year in range(YEAR_START,YEAR_END):
            print('Year: ' + str(year))
            df_year_path = path_to_df + str(year)
            embedding_year_path = path_to_embeddings + str(year)

            df_paths_per_month = [os.path.join(df_year_path, f) for f in os.listdir(df_year_path) if os.path.isfile(os.path.join(df_year_path, f))]
            print(df_paths_per_month)

            embedding_paths_per_month = [os.path.join(embedding_year_path, f) for f in os.listdir(embedding_year_path) if os.path.isfile(os.path.join(embedding_year_path, f))]
            print(embedding_paths_per_month)


            rand_int = random.randint(0,len(df_paths_per_month)-1)
            val_df = df_paths_per_month[rand_int]
            df_paths_per_month.remove(val_df)
            val_embeddings_dataset = embedding_paths_per_month[rand_int]
            embedding_paths_per_month.remove(val_embeddings_dataset)


            with open(val_df, 'rb') as f:
                data_frame = pickle.loads(f.read())
            Y_val = data_frame['AAPL'].values
            print(Counter(list(Y_val)))
            with open(val_embeddings_dataset, 'rb') as f:
                embedded_words = pickle.loads(f.read())
            X_val = np.array(embedded_words)
            print(X_val.shape)
            assert (len(data_frame) == len(embedded_words))
            print('VALIDATION SET CREATED')

            for data_frame_path, embedding_path in zip(df_paths_per_month, embedding_paths_per_month):

                with open(data_frame_path, 'rb') as f:
                    data_frame = pickle.loads(f.read())

                Y = data_frame['AAPL'].values
                print(Counter(list(Y)))

                with open(embedding_path, 'rb') as f:
                    embedded_words = pickle.loads(f.read())

                X = np.array(embedded_words)
                print(X.shape)

                assert(len(data_frame) == len(embedded_words))


                print('Training on' + data_frame_path + ' with ' + str(len(Y)) + ' samples')

                model.fit(X, Y, batch_size=512, validation_split=0.1, verbose=2)
                model.save('stock_pred_model.h5')

                predictions = model.predict(X_val, batch_size=512)
                rounded_predicitons = np.round(predictions)
                print(Counter(rounded_predicitons))
                # print(metrics.precision_recall_fscore_support(Y_val, rounded_predicitons))

# create_embedded_headdings()
train()


