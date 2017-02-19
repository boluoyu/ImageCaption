# coding: utf-8
from __future__ import print_function

"""
Using keras to caption image
Reference:
https://github.com/fchollet/keras/issues/2295
https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
"""

import os
import sys
import json
import pickle
from collections import OrderedDict, Counter

# os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['KERAS_BACKEND'] = 'theano'

import nltk
from nltk.tokenize import word_tokenize
from PIL import Image
from scipy.misc import imresize
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding, GRU, LSTM, TimeDistributed, RepeatVector, Merge
from keras.layers import Dense, Activation, Dropout, Bidirectional
from keras.optimizers import Nadam, SGD, RMSprop, Adam
from sklearn.utils import shuffle

# Download tokenizer models if needed
nltk.download('punkt')

# Random seed
np.random.seed(0)

# Read image file_name and their captions
file_name_caption = pickle.load(open( "file_name_caption.bat", "rb" ))
# Only train
file_name_caption = {k: v for k, v in file_name_caption.items() if 'train' in k}
# Read image file_name and their VGG 16 weights
file_name_images = pickle.load(open( "file_name_images.bat", "rb" ))

train_size = len(file_name_caption)
print('train_size', train_size)

START = '<start>'
END = '<end>'
UNK = '<unk>'
PAD = '<pad>'

min_count = 2
max_len = 0
train_words_size = 0
vocabulary = Counter()
for caption in tqdm(file_name_caption.values(), file=sys.stdout, total=len(file_name_caption)):
    sent = word_tokenize(caption)
    vocabulary.update(sent)
    train_words_size += len(sent)
    if len(sent) > max_len: max_len = len(sent)

vocabulary = [k for k, v in vocabulary.items() if v >= min_count]
vocabulary = sorted(list(set(vocabulary)))
word_index = OrderedDict()
index_word = OrderedDict()
for index, word in enumerate([START, END, UNK, PAD] + vocabulary):
    word_index[word] = index
    index_word[index] = word
vocabulary_size = len(word_index)

print('vocabulary_size', vocabulary_size)
print('max_len', max_len)
print('train_words_size', train_words_size)

pickle.dump({
    'word_index': word_index,
    'index_word': index_word,
    'max_len': max_len,
    'vocabulary_size': vocabulary_size,
    'vocabulary': vocabulary,
    'START': START,
    'END': END,
    'UNK': UNK,
    'PAD': PAD
}, open('argument.dat', 'wb'), protocol=2)

batch_size = 256
embedding_size = 256
samples_per_epoch = int(train_words_size / batch_size + 1) * batch_size
file_name_caption_list = list(file_name_caption.items())
file_name_caption_list = sorted(file_name_caption_list, key=lambda x: x[0])
file_name_caption_list = shuffle(file_name_caption_list, random_state=0)

# Image model
image_model = Sequential()
image_model.add(Dense(256, activation='relu', input_shape=(4096,)))
image_model.add(RepeatVector(1))

language_model = Sequential()
language_model.add(Embedding(vocabulary_size, embedding_size, input_length=max_len))
language_model.add(LSTM(256, return_sequences=True))
language_model.add(TimeDistributed(Dense(256)))

model = Sequential()
model.add(Merge([image_model, language_model], mode='concat', concat_axis=1))
model.add(Bidirectional(LSTM(256, return_sequences=False)))
model.add(Dense(vocabulary_size))
model.add(Activation('softmax'))

optimizer = Adam(lr=0.001, clipnorm=5.)
# optimizer = Adam(lr=0.001, clipvalue=0.5)
# optimizer = Adam(lr=0.001)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

def data_flow(file_name_caption_list, file_name_images, word_index, max_len, vocabulary_size, batch_size):
    X_img = []
    X_lang = []
    Y = []
    while True:
        for file_name, caption in file_name_caption_list:
            if file_name in file_name_images:
                img_weights = file_name_images[file_name]
                sent = [START] + word_tokenize(caption) + [END]
                for i in range(1, len(sent)):

                    if len(X_img) == batch_size:
                        X_img = np.asarray(X_img)
                        X_lang = np.asarray(X_lang)
                        Y = np.asarray(Y).reshape([batch_size, vocabulary_size])
                        yield [X_img, X_lang], Y
                        X_img = []
                        X_lang = []
                        Y = []

                    input_sent = sent[:i]
                    padding_size = max_len - len(input_sent)

                    input_sent = input_sent + padding_size * [PAD]
                    input_sent_index = []
                    for w in input_sent:
                        if w in word_index:
                            input_sent_index.append(word_index[w])
                        else:
                            input_sent_index.append(word_index[UNK])

                    if i < len(sent):
                        target_word = sent[i]
                    else:
                        target_word = END

                    if target_word in word_index:
                        target_word_index = word_index[target_word]
                    else:
                        target_word_index = word_index[UNK]

                    if len(input_sent_index) == max_len:

                        X_img.append(img_weights)
                        X_lang.append(input_sent_index)
                        y = np.zeros([vocabulary_size])
                        y[target_word_index] = 1.0
                        Y.append([y])
# Test data flow
i = 0
for [x_img, x_lang], y in data_flow(file_name_caption_list, file_name_images, word_index, max_len, vocabulary_size, batch_size):
    print(x_img.shape, x_lang.shape, y.shape)
    i += 1
    if i > 10: break

# Fit coco train data
model.fit_generator(
    data_flow(file_name_caption_list, file_name_images, word_index, max_len, vocabulary_size, batch_size),
    samples_per_epoch=samples_per_epoch,
    nb_epoch=20
)

model.save('keras_image_caption_model.dat')
