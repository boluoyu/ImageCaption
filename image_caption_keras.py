#!/usr/bin/env python3
# coding: utf-8

"""
Using keras to caption image
Reference:
https://github.com/fchollet/keras/issues/2295
https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
"""

import os
import sys
import json
from collections import OrderedDict, Counter

import nltk
from nltk.tokenize import word_tokenize
from PIL import Image
from scipy.misc import imresize
import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Embedding, GRU, TimeDistributed, RepeatVector, Merge
from keras.layers import Dense, Activation
from keras.optimizers import Nadam
from sklearn.utils import shuffle

from vgg16 import VGG_16

# Download tokenizer models if needed
nltk.download('punkt')

# Random seed
np.random.seed(0)

# Path to dataset
vgg_model_weights = '/home/qhduan/Downloads/COCO/vgg16_weights.h5'
coco_train = '/home/qhduan/Downloads/COCO/train2014'
coco_caption = '/home/qhduan/Downloads/COCO/annotations/captions_train2014.json'

def get_file_name_caption(coco_caption):
    captions = json.load(open(coco_caption, 'r'))
    image_id_caption = {}
    for caption in captions['annotations']:
        image_id_caption[caption['image_id']] = caption['caption']
    ret = OrderedDict()
    for img in captions['images']:
        if img['id'] in image_id_caption:
            caption = image_id_caption[img['id']]
            file_name = img['file_name']
            ret[file_name] = caption
    return ret

# Read image file_name and it's caption
file_name_caption = get_file_name_caption(coco_caption)

train_size = len(file_name_caption)
print('train_size', train_size)

START = '<start>'
END = '<end>'
UNK = '<unk>'
PAD = '<pad>'

min_count = 3
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

batch_size = 8
embedding_size = 128
rnn_size = 256
model_output = 256
samples_per_epoch = int(train_words_size / batch_size + 1) * batch_size
file_name_caption_list = list(file_name_caption.items())
file_name_caption_list = sorted(file_name_caption_list, key=lambda x: x[0])
file_name_caption_list = shuffle(file_name_caption_list, random_state=0)

# VGG 16 model with pretrained weights
vgg_model = VGG_16()
vgg_model.load_weights(vgg_model_weights)
vgg_model.layers.pop()
vgg_model.layers.pop()
vgg_model.trainable = False

# Image model
image_model = Sequential()
image_model.add(vgg_model)
image_model.add(Dense(model_output, activation='relu'))
image_model.add(RepeatVector(max_len))

language_model = Sequential()
language_model.add(Embedding(vocabulary_size, embedding_size, input_length=max_len))
language_model.add(GRU(rnn_size, return_sequences=True))
language_model.add(TimeDistributed(Dense(model_output)))

model = Sequential()
model.add(Merge([image_model, language_model], mode='concat', concat_axis=1))
model.add(GRU(rnn_size, return_sequences=False))
model.add(Dense(vocabulary_size))
model.add(Activation('softmax'))

optimizer = Nadam(lr=0.0001, clipnorm=1., clipvalue=5.)
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

def data_flow(file_name_caption_list, word_index, coco_dir, max_len, vocabulary_size, batch_size):
    X_img = []
    X_lang = []
    Y = []
    while True:
        for file_name, caption in file_name_caption_list:
            path = os.path.join(coco_dir, file_name)
            if os.path.exists(path):
                img = np.array(Image.open(path))
                if len(img.shape) == 3:
                    img = imresize(img, (224, 224))
                    img = img.astype(np.float32)
                    img[:,:,0] -= 103.939
                    img[:,:,1] -= 116.779
                    img[:,:,2] -= 123.68
                    img = img.transpose([2, 0, 1])
                    sent = [START] + word_tokenize(caption) + [END]
                    for i in range(1, len(sent) - 1):
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

                        target_word = sent[i]
                        if target_word in word_index:
                            target_word_index = word_index[target_word]
                        else:
                            target_word_index = word_index[UNK]

                        X_img.append(img)
                        X_lang.append(input_sent_index)
                        y = np.zeros([vocabulary_size])
                        y[target_word_index] = 1.0
                        Y.append([y])
# Test data flow
i = 0
for [x_img, x_lang], y in data_flow(file_name_caption_list, word_index, coco_train, max_len, vocabulary_size, batch_size):
    print(x_img.shape, x_lang.shape, y.shape)
    i += 1
    if i > 10: break

# Fit coco train data
model.fit_generator(
    data_flow(file_name_caption_list, word_index, coco_train, max_len, vocabulary_size, batch_size),
    samples_per_epoch=samples_per_epoch,
    nb_epoch=5
)

model.save('keras_image_caption_model.dat')
