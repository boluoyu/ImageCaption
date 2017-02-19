
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

```
$ python3 image_caption_keras.py
Using Theano backend.
Using gpu device 0: GeForce GTX 1070 (CNMeM is disabled, cuDNN 5105)
/home/qhduan/.local/lib/python3.5/site-packages/theano/sandbox/cuda/__init__.py:600: UserWarning: Your cuDNN version is more recent than the one Theano officially supports. If you see any problems, try updating Theano or downgrading cuDNN to version 5.
  warnings.warn(warn)
[nltk_data] Downloading package punkt to /home/qhduan/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
train_size 82783
100%|████████████████████████████████████████████████████████████████████████████████████████| 82783/82783 [00:06<00:00, 12363.22it/s]
vocabulary_size 8679
max_len 55
train_words_size 935568
(256, 4096) (256, 55) (256, 8679)
(256, 4096) (256, 55) (256, 8679)
(256, 4096) (256, 55) (256, 8679)
(256, 4096) (256, 55) (256, 8679)
(256, 4096) (256, 55) (256, 8679)
(256, 4096) (256, 55) (256, 8679)
(256, 4096) (256, 55) (256, 8679)
(256, 4096) (256, 55) (256, 8679)
(256, 4096) (256, 55) (256, 8679)
(256, 4096) (256, 55) (256, 8679)
(256, 4096) (256, 55) (256, 8679)
Epoch 1/20
935680/935680 [==============================] - 980s - loss: 4.1896 - acc: 0.2899          
Epoch 2/20
935680/935680 [==============================] - 980s - loss: 3.1131 - acc: 0.4088     
Epoch 3/20
935680/935680 [==============================] - 1024s - loss: 2.8843 - acc: 0.4295    
Epoch 4/20
935680/935680 [==============================] - 1053s - loss: 2.7526 - acc: 0.4415     
Epoch 5/20
935680/935680 [==============================] - 1053s - loss: 2.6622 - acc: 0.4504     
Epoch 6/20
935680/935680 [==============================] - 1001s - loss: 2.5747 - acc: 0.4587     
Epoch 7/20
935680/935680 [==============================] - 988s - loss: 2.4988 - acc: 0.4663      
Epoch 8/20
935680/935680 [==============================] - 1060s - loss: 2.4339 - acc: 0.4740     
Epoch 9/20
935680/935680 [==============================] - 1032s - loss: 2.3833 - acc: 0.4802     
Epoch 10/20
935680/935680 [==============================] - 1005s - loss: 2.3305 - acc: 0.4866     
Epoch 11/20
935680/935680 [==============================] - 1007s - loss: 2.2816 - acc: 0.4927     
Epoch 12/20
935680/935680 [==============================] - 1063s - loss: 2.2408 - acc: 0.4987     
Epoch 13/20
935680/935680 [==============================] - 1031s - loss: 2.1983 - acc: 0.5048     
Epoch 14/20
935680/935680 [==============================] - 995s - loss: 2.1705 - acc: 0.5086      
Epoch 15/20
935680/935680 [==============================] - 991s - loss: 2.1432 - acc: 0.5122     
Epoch 16/20
935680/935680 [==============================] - 984s - loss: 2.1109 - acc: 0.5173      
Epoch 17/20
935680/935680 [==============================] - 978s - loss: 2.0837 - acc: 0.5222      
Epoch 18/20
935680/935680 [==============================] - 978s - loss: 2.0582 - acc: 0.5257     
Epoch 19/20
935680/935680 [==============================] - 978s - loss: 2.0487 - acc: 0.5281     
Epoch 20/20
935680/935680 [==============================] - 978s - loss: 2.0179 - acc: 0.5325
```
