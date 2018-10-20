from keras.layers import Input, Dense, LSTM, Dropout
from keras.engine import Model
from keras.optimizers import adam
from keras import metrics

def get_stock_pred_model(batch_size, sentence_shape):

    sentence_input = Input(batch_shape=(batch_size, sentence_shape[0], sentence_shape[1]), name='sentence_input')

    sentence_lstm1 = LSTM(1024, return_sequences=False, activation='relu', name='sentence_lstm')(sentence_input)

    sentence_dropout1 = Dropout(0.4, name='sentence_dropout_1')(sentence_lstm1)

    dense = Dense(1024, activation='relu', name='image_sentence_dense')(sentence_dropout1)

    sentence_dropout2 = Dropout(0.4, name='image_sentence_dropout_2')(dense)

    dense2 = Dense(512, activation='relu', name='image_sentence_dense_2')(sentence_dropout2)

    activation = Dense(1, activation='sigmoid', name='activation_dense')(dense2)

    model = Model([sentence_input], activation)

    ad = adam(lr=0.0002)
    model.compile(optimizer=ad, loss='binary_crossentropy', metrics=['accuracy'])

    return model
