# coding: utf-8
from __future__ import print_function

"""
Using VGG 16 pretrained model to generate COCO images' weights
"""

import os
import sys
import json
import pickle
from collections import OrderedDict
import numpy as np
from PIL import Image
from scipy.misc import imresize
from tqdm import tqdm
from vgg16 import VGG_16

vgg_model_weights = '/home/qhduan/Downloads/COCO/vgg16_weights.h5'
coco_train = '/home/qhduan/Downloads/COCO/train2014'
coco_train_caption = '/home/qhduan/Downloads/COCO/annotations/captions_train2014.json'
coco_val = '/home/qhduan/Downloads/COCO/val2014'
coco_val_caption = '/home/qhduan/Downloads/COCO/annotations/captions_val2014.json'

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
train_file_name_caption = get_file_name_caption(coco_train_caption)
val_file_name_caption = get_file_name_caption(coco_val_caption)

file_name_caption = {}
for x, y in train_file_name_caption.items():
    file_name_caption[x] = y
for x, y in val_file_name_caption.items():
    file_name_caption[x] = y

# VGG 16 model with pretrained weights
vgg_model = VGG_16()
vgg_model.load_weights(vgg_model_weights)
vgg_model.layers.pop()
vgg_model.layers.pop()
vgg_model.outputs = [vgg_model.layers[-1].output]
vgg_model.layers[-1].outbound_nodes = []
vgg_model.trainable = False

batch_size = 16

file_name_images = {}
X = []

for file_name in tqdm(train_file_name_caption.keys(), file=sys.stdout, total=len(train_file_name_caption)):
    path = os.path.join(coco_train, file_name)
    img = np.array(Image.open(path))
    if len(img.shape) == 3:
        img = imresize(img, (224, 224))
        img = img.astype(np.float32)
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
        img = img.transpose([2, 0, 1])
        X.append((file_name, img))
        if len(X) == batch_size:
            ret = vgg_model.predict(np.asarray([x[1] for x in X]))
            for index, x in enumerate(X):
                file_name_images[x[0]] = ret[index]
            X = []

if len(X) > 0:
    ret = vgg_model.predict(np.asarray([x[1] for x in X]))
    for index, x in enumerate(X):
        file_name_images[x[0]] = ret[index]
    X = []

for file_name in tqdm(val_file_name_caption.keys(), file=sys.stdout, total=len(val_file_name_caption)):
    path = os.path.join(coco_val, file_name)
    img = np.array(Image.open(path))
    if len(img.shape) == 3:
        img = imresize(img, (224, 224))
        img = img.astype(np.float32)
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
        img = img.transpose([2, 0, 1])
        X.append((file_name, img))
        if len(X) == batch_size:
            ret = vgg_model.predict(np.asarray([x[1] for x in X]))
            for index, x in enumerate(X):
                file_name_images[x[0]] = ret[index]
            X = []

if len(X) > 0:
    ret = vgg_model.predict(np.asarray([x[1] for x in X]))
    for index, x in enumerate(X):
        file_name_images[x[0]] = ret[index]
    X = []

pickle.dump(file_name_images, open( "file_name_images.bat", "wb" ), protocol=2)
pickle.dump(file_name_caption, open( "file_name_caption.bat", "wb" ), protocol=2)
