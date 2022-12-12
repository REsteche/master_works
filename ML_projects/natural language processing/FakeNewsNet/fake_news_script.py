# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 15:37:55 2022

@author: ruben
"""

# data analysis libs
import numpy as np
import pandas as pd

# visualizatioon libs
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning libs
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# tensorFlow & Keras
from tensorflow import keras as kr
from tqdm.keras import TqdmCallback


# Data Loading
file = 'FakeNewsNet.csv'
columns = ['title','real']
df = pd.read_csv(file, usecols=columns)
print(df.head())

# drop NULLs (if any) & reset index
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

# adding a new column [word count]
df['wcount'] = df['title'].apply(lambda x: len(x.split(' ')))

# view
print(f"After cleaning we have {df.shape[0]} news...")

# True && False visualization - counting fake and real
count = []
for i in df['real'].unique():
    count.append(df[df['real'] == i].count()[1])
    
# Histogram counting barplot for fake&realnews
g = sns.catplot(
    data=df, kind="bar",
    x=df['real'].unique(), y=count,alpha=.6,
    palette="colorblind")

#g.despine(left=True)
g.set_axis_labels("", "count")

# feature engineering & training slipt
x = df['title'].values # feature
y = df['real'].values # target

# train & validation split [80-20]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state= np.random.randint(10))

# split visualization
print(f"Training records: {x_train.shape[0]} | Testing records: {x_test.shape[0]}")

# text pre-processing
tok = kr.preprocessing.text.Tokenizer(num_words=None,
                                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                      lower=True,
                                      split=' ',
                                      char_level=False,
                                      oov_token=None)
# fit 
tok.fit_on_texts(x_train)

# text tokenizing
# tokenized text data
tok_train = tok.texts_to_sequences(x_train)
tok_test = tok.texts_to_sequences(x_test)

# Pad the sequences for the training 
max_length = int(df.wcount.quantile(0.75))   # taking the 75th percentile of word count

padded_train = kr.preprocessing.sequence.pad_sequences(tok_train, maxlen=max_length, padding='post')
padded_test = kr.preprocessing.sequence.pad_sequences(tok_test, maxlen=max_length, padding='post')