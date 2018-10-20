from keras.layers import Input, Dense, LSTM, Dropout, concatenate
from keras.engine import Model
from keras.optimizers import adam
from keras import metrics

def get_stock_pred_model(batch_size, headder_length, sentence_length):

    headder_input = Input(batch_shape=(batch_size, headder_length, 300), name='headder_input')

    headder_lstm1 = LSTM(512, return_sequences=False, activation='relu', name='headder_lstm')(headder_input)

    sentence_input = Input(batch_shape=(batch_size, sentence_length, 300), name='sentence_input')

    sentence_lstm1 = LSTM(512, return_sequences=False, activation='relu', name='sentence_lstm')(sentence_input)

    concat = concatenate([headder_lstm1, sentence_lstm1])

    sentence_dropout1 = Dropout(0.2, name='sentence_dropout_1')(concat)

    dense = Dense(1024, activation='relu', name='image_sentence_dense')(sentence_dropout1)

    sentence_dropout2 = Dropout(0.2, name='image_sentence_dropout_2')(dense)

    dense2 = Dense(512, activation='relu', name='image_sentence_dense_2')(sentence_dropout2)

    sentence_dropout3 = Dropout(0.2, name='image_sentence_dropout_3')(dense2)

    dense3 = Dense(128, activation='relu', name='image_sentence_dense_3')(sentence_dropout3)

    activation = Dense(1, activation='sigmoid', name='activation_dense')(dense3)

    model = Model([headder_input, sentence_input], activation)

    ad = adam(lr=0.0005)
    model.compile(optimizer=ad, loss='binary_crossentropy', metrics=['accuracy'])

    return model
