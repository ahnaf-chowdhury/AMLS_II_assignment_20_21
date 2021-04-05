import os
import re
import pandas as pd
import nltk
import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
nltk.download('stopwords')
from nltk.corpus import stopwords

def text_cleaning(text):
    # cleans phrases. e.g. removes punctuations, changes all letters to lowercase,
    # removes stop words

    # stopwords do not usually add value to sentiment analysis and can be removed
    stop_words = set(stopwords.words('english'))

    # the nltk stopwords set includes negating words, which should be removed from
    #the stopwords set
    neg = ["aren't", "didn't", "don't", "doesn't", "hadn't",  "hasn't", "haven't", "isn't", "no", "not", "shouldn't", "wasn't", "weren't", "wouldn't", "couldn't",
          "aren", "didn", "don", "doesn", "hadn", "hasn", "haven", "isn", "shouldn", "wasn", "weren", "wouldn", "couldn",
           "mightn't", "mightn", "mustn't", "mustn", "needn't", "needn", "shan't", "shan", "won't", "won"]

    stop_words.difference_update(neg)

    if text:
        text = ' '.join(text.split('.'))
        text = re.sub('\/', ' ', text)
        text = re.sub(r'\\', ' ', text)
        text = re.sub(r'((http)\S+)', '', text)
        text = re.sub(r'\s+', ' ', re.sub('[^A-Za-z]', ' ', text.strip().lower())).strip()
        text = re.sub(r'\W+', ' ', text.strip().lower()).strip()

        text_final = [word for word in text.split() if word not in stop_words]

        if len(text_final) == 0:       # if all words are irrelevant, we may still wish to include the phrase
            text_final = [word for word in text.split()]
        return " ".join(text_final)
    return ""

def save_history(filename, training_time, history_dict):
    # saves history in a csv file from a dictionary
    df_history = pd.DataFrame.from_dict(history_dict)
    df_history['training_time'] = training_time

    # create directory "history" if it doesn't exist:
    try_mkdir('./history')

    path = os.path.join('./history', filename)
    df_history.to_csv(path)

def try_mkdir(dir):
    #creating a directory with exception handling

    try:
        os.mkdir(dir)
    except OSError:
        print("Creation of the directory %s failed" % dir)
    else:
        print("Successfully created the directory %s " % dir)

def get_prediction(arr):
    # function holder
      return arr.argmax()

def get_arg_use_pretrained():
    # get boolean variable on whether to use a pretrained model or not.
    # By default, a model is trained instead.
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pretrained", type=bool, default=False, help="Whether to use a pretrained model or not. Pass 'True' to use pretrained model.")
    return vars(ap.parse_args())['pretrained']
