# coding: utf-8
from __future__ import print_function

"""
Image Caption Web Service
"""

import os
import re
import base64
import pickle
from io import BytesIO

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
IMAGE_PATTERN =  pattern = re.compile(
    '^.*\.(png|jpeg|gif|bmp|jpg)$',
    re.IGNORECASE
)

from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model

from vgg16 import VGG_16

vgg_model_weights = 'vgg16_weights.h5'
keras_model = 'keras_image_caption_model.dat'

print('Loading VGG 16 model')
# VGG 16 model with pretrained weights
vgg_model = VGG_16()
vgg_model.load_weights(vgg_model_weights)
vgg_model.layers.pop()
vgg_model.layers.pop()
vgg_model.outputs = [vgg_model.layers[-1].output]
vgg_model.layers[-1].outbound_nodes = []
vgg_model.trainable = False

print('Loading keras model')
model = load_model(keras_model)

print('Loading arguments')
argument = pickle.load(open('argument.dat', 'rb'))
word_index = argument['word_index']
index_word = argument['index_word']
max_len = argument['max_len']
START = argument['START']
END = argument['END']
UNK = argument['UNK']
PAD = argument['PAD']

def sent_to_index(input_sent, word_index, max_len):
    padding_size = max_len - len(input_sent)

    input_sent = input_sent + padding_size * [PAD]
    input_sent_index = []
    for w in input_sent:
        if w in word_index:
            input_sent_index.append(word_index[w])
        else:
            input_sent_index.append(word_index[UNK])
    return input_sent_index

def predict_weights(weights, word_index, index_word, max_len):
    sent = [START]
    while True:
        sent_index = sent_to_index(sent, word_index, max_len)
        index = model.predict([np.asarray([weights]), np.asarray([sent_index])]).argmax()
        if index in index_word:
            word = index_word[index]
        else:
            word = UNK
        sent.append(word)
        if word == END or len(sent) > max_len:
            break
    return sent

app = Flask(__name__)

TEMPLATE = '''
<!doctype html>
<html>
    <head>
        <title>Image Caption</title>
    </head>
    <body>
        <h1>Upload new Image</h1>
        <form method=post enctype=multipart/form-data>
            <p>
                <input type=file name=file>
                <input type=submit value=Caption>
            </p>
        </form>
        {append}
    </body>
</html>
'''

ERROR = '''
<div>
    <h1>ERROR</h1>
    <h2>{}</h2>
</div>
'''

RESULT = '''
<div>
    <h2>Last Result</h2>
    <img src='data:image/png;base64,{base64}' />
    <h3>{caption}</h3>
</div>
'''

def send_template(append=''):
    return TEMPLATE.format(append=append)

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return send_template(ERROR.format('No file'))
        file = request.files['file']
        if IMAGE_PATTERN.match(file.filename) is None:
            return send_template(
                ERROR.format('No an image, accept jpg/png/gif')
            )
        try:
            img = Image.open(file)
            img = img.resize((224, 224))
            img = img.convert('RGB')
            imgfile = BytesIO()
            img.save(imgfile, format='PNG')
            encoded_string = base64.b64encode(imgfile.getvalue())
        except:
            return send_template(
                ERROR.format('No a valid image')
            )
        try:
            weights = vgg_model.predict(
                np.array(img).transpose([2, 0, 1]).reshape([1, 3, 224, 224])
            )[0]
            ret = predict_weights(weights, word_index, index_word, max_len)
            while ret[0] == START:
                ret = ret[1:]
            while ret[-1] == END:
                ret = ret[:-1]
        except:
            return send_template(
                ERROR.format('Failed to predict')
            )
        return send_template(RESULT.format(
            base64=encoded_string,
            caption=' '.join(ret)
        ))
    return send_template()

if __name__ == '__main__':
    print('Start web service')
    app.run(host='0.0.0.0', port=10030)
