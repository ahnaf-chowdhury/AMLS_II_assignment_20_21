import ahnaf_nlp
import os
import pandas as pd
import nltk
import time
import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.preprocessing.text import one_hot

# nltk.download('stopwords')
# from nltk.corpus import stopwords

def get_argmax(arr):
    # returns the index of the largest number in an array
    return arr.argmax()

if __name__ == '__main__':

    TRAIN_MODEL = not ahnaf_nlp.get_arg_use_pretrained()
                        # If true, a model is trained (default),
                        # and if false, a pretrained model is loaded.
                        # argument is parsed from the command line
    PT_MODEL_PATH = './A/models/lstm-2-apr-b512-dropout0-2-bn-dim128-emb-untr-epochs-335-2-layer-ttv-final.h5'
                        # path where the pretrained model is stored
    PT_MODEL_HISTORY_PATH = './A/models/history-lstm-2-apr-b512-dropout0-2-bn-dim128-emb-untr-epochs-335-2-layer-ttv-final.csv'

    #-------------------------------------------------------------------------------
    # Data preprocessing
    #-------------------------------------------------------------------------------

    print('Loading Data ...')

    # loading data:
    data_dir_train = './Datasets/train.tsv'
    data_dir_test = './Datasets/test.tsv'

    data_train = pd.read_csv(data_dir_train, sep='\t')
    data_test = pd.read_csv(data_dir_test, sep='\t')

    print('Loading complete.')

    # cleaning the data:
    # (e.g. removing punctuations, changing all letters to lowercase, removing stop words)

    print('Preprocessing data ...')

    data_train['Phrase'] = data_train['Phrase'].apply(lambda x: ahnaf_nlp.text_cleaning(x))
    data_test['Phrase'] = data_test['Phrase'].apply(lambda x: ahnaf_nlp.text_cleaning(x))

    # adding a 'length' column to filter out phrases that only consist of punctuation marks
    # which appear as blank strings after cleaning

    data_train['Length'] = data_train['Phrase'].apply(lambda x: len(x.split()))
    data_test['Length'] = data_test['Phrase'].apply(lambda x: len(x.split()))

    # train-test-validation split

    validation_split = 0.15
    test_split = 0.15

    x_train, x_test, y_train, y_test = train_test_split(
        data_train[data_train['Length']>0]['Phrase'],
        data_train[data_train['Length']>0]['Sentiment'],
        test_size = test_split,
        stratify = data_train[data_train['Length'] > 0]['Sentiment'], # ('Length']>0 filters out 'empty' phrases, which were previously punctuation marks)
        random_state = 40)

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train,
        y_train,
        test_size = validation_split,
        stratify = y_train,
        random_state = 40)

    # encoding the text using 'one_hot' from keras
    # (each word is encoded as a number upto 20000)

    vocab_size = 20000

    encodings_train = x_train.apply(lambda z: one_hot(z, vocab_size))
    encodings_validation = x_validation.apply(lambda z: one_hot(z, vocab_size))
    encodings_test = x_test.apply(lambda z: one_hot(z, vocab_size))

    # padding the encodings to the same length

    maxlen = 200    #default

    max_len_in_train = len(max(encodings_train, key = lambda i: len(i)))                #
    max_len_in_validation = len(max(encodings_validation, key = lambda i: len(i)))      #
    max_len_in_test = len(max(encodings_test, key = lambda i: len(i)))                  # If the longest encoding
    max_len_in_dataset = max(max_len_in_train, max_len_in_validation, max_len_in_test)  # in the dataset has a length
                                                                                        # that is shorter than the
    if max_len_in_dataset < maxlen:                                                     # default length (200), update
        maxlen = max_len_in_dataset                                                     # maxlen

    encodings_train = keras.preprocessing.sequence.pad_sequences(encodings_train, maxlen=maxlen)
    encodings_validation = keras.preprocessing.sequence.pad_sequences(encodings_validation, maxlen=maxlen)
    encodings_test = keras.preprocessing.sequence.pad_sequences(encodings_test, maxlen=maxlen)

    # batched train and validation datasets:

    batch_size = 512
    n_epochs = 2
    embedding_dims = 128

    train_dataset = tf.data.Dataset.from_tensor_slices((
        encodings_train, y_train.values)).shuffle(10000).batch(batch_size).repeat()

    val_dataset = tf.data.Dataset.from_tensor_slices((
        encodings_validation, y_validation.values)).shuffle(10000).batch(batch_size)

    print('Preprocessing complete.')

    #-------------------------------------------------------------------------------
    # Building and training (or loading) the LSTM
    #-------------------------------------------------------------------------------

    if TRAIN_MODEL:
        print('Building and training LSTM ...')
        model = Sequential()
        inputs = keras.Input(shape=(None,), dtype="int32")

        model.add(inputs)
        model.add(Embedding(vocab_size, embedding_dims, trainable=False))

        model.add(Bidirectional(LSTM(embedding_dims//2, return_sequences=True, activation='softmax')))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Bidirectional(LSTM(embedding_dims//2)))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        model.add(Dense(5, activation="sigmoid"))

        model.summary()

        model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        training_st = time.time()

        history = model.fit(train_dataset,
                            batch_size=batch_size,
                            epochs=n_epochs,
                            validation_steps = len(x_validation) // batch_size,
                            steps_per_epoch = len(x_train) // batch_size,
                            validation_data = val_dataset)

        training_time = time.time() - training_st

        history = pd.DataFrame.from_dict(history.history)

        print('Training complete.')

    # (or) Loading the LSTM from a '.h5' file (and loading the stored history of training)
    else:
        print('Loading pretrained LSTM')
        model = keras.models.load_model(PT_MODEL_PATH)
        history = pd.read_csv(PT_MODEL_HISTORY_PATH)

        training_time = history['training_time'].iloc[0]
        n_epochs = len(history.index)

        print('Model loaded.')

    #-------------------------------------------------------------------------------
    # Evaluating the LSTM by running it on the test set
    #-------------------------------------------------------------------------------

    print('Running model on test set.')

    test_st = time.time()
    y_test_predicted_np = model.predict_on_batch(encodings_test)  # numpy array
    test_time = time.time() - test_st

    y_test_predicted = np.apply_along_axis(get_argmax, 1, y_test_predicted_np)

    acc_test = accuracy_score(y_test, y_test_predicted)

    acc_train = history['accuracy'].iloc[n_epochs - 1]

    acc_val = history['val_accuracy'].iloc[n_epochs - 1]

    print('Process complete.\n\n')

    #-------------------------------------------------------------------------------

    print('Accuracy (train, validation, test):{}, {}, {};'.format(acc_train, acc_val, acc_test))
    if TRAIN_MODEL:
        print('Time taken to train {} epochs: {} s (trained during this session)'.format(n_epochs, training_time))
    else:
        print('Time taken to train {} epochs: {} s (trained earlier using Google Colab TPUs)'.format(n_epochs, training_time))

    print('Time taken to run the model on the test set of size {}: {} s'.format(len(y_test), test_time))
