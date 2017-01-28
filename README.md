
# Image Caption


You could download Microsoft COCO dataset from [here](http://mscoco.org/dataset/#download)

VGG 16 pretrained model could download from [here](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view)

如果你是中国人，可以从百度网盘的[这里](https://pan.baidu.com/s/1eSLwvFc)下载

Then unzip COCO and specify path in image_caption_keras.py :

```
vgg_model_weights = '/home/qhduan/Downloads/COCO/vgg16_weights.h5'
coco_train = '/home/qhduan/Downloads/COCO/train2014'
coco_caption = '/home/qhduan/Downloads/COCO/annotations/captions_train2014.json'
```

You could open *preview.ipynb* to view the train result and test result.

qhduan@qhduan-memect:~/Downloads/COCO$ python3 ./image_caption_keras.py
Using Theano backend.
Using gpu device 0: GeForce GTX 960M (CNMeM is disabled, cuDNN 5105)
/home/qhduan/.local/lib/python3.5/site-packages/theano/sandbox/cuda/__init__.py:600: UserWarning: Your cuDNN version is more recent than the one Theano officially supports. If you see any problems, try updating Theano or downgrading cuDNN to version 5.
  warnings.warn(warn)
[nltk_data] Downloading package punkt to /home/qhduan/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
train_size 82783
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 82783/82783 [00:07<00:00, 10523.32it/s]
vocabulary_size 6627
max_len 55
train_words_size 935568
(256, 4096) (256, 55) (256, 6627)
(256, 4096) (256, 55) (256, 6627)
(256, 4096) (256, 55) (256, 6627)
(256, 4096) (256, 55) (256, 6627)
(256, 4096) (256, 55) (256, 6627)
(256, 4096) (256, 55) (256, 6627)
(256, 4096) (256, 55) (256, 6627)
(256, 4096) (256, 55) (256, 6627)
(256, 4096) (256, 55) (256, 6627)
(256, 4096) (256, 55) (256, 6627)
(256, 4096) (256, 55) (256, 6627)
Epoch 1/10
935680/935680 [==============================] - 3128s - loss: 4.1925 - acc: 0.2965         
Epoch 2/10
935680/935680 [==============================] - 3060s - loss: 3.4850 - acc: 0.3683     
Epoch 3/10
935680/935680 [==============================] - 3058s - loss: 3.2198 - acc: 0.3910     
Epoch 4/10
935680/935680 [==============================] - 3102s - loss: 3.0703 - acc: 0.4058     
Epoch 5/10
935680/935680 [==============================] - 3085s - loss: 2.9631 - acc: 0.4167     
Epoch 6/10
935680/935680 [==============================] - 3130s - loss: 2.8886 - acc: 0.4252     
Epoch 7/10
935680/935680 [==============================] - 3049s - loss: 2.8271 - acc: 0.4316     
Epoch 8/10
935680/935680 [==============================] - 3051s - loss: 2.7768 - acc: 0.4377     
Epoch 9/10
935680/935680 [==============================] - 3048s - loss: 2.7363 - acc: 0.4419     
Epoch 10/10
935680/935680 [==============================] - 3048s - loss: 2.7039 - acc: 0.4459
