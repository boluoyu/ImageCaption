
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

qhduan@qhduan-memect:~/Downloads/COCO$ ve/bin/python image_caption_keras.py
Using Theano backend.
Using gpu device 0: GeForce GTX 960M (CNMeM is disabled, cuDNN 5105)
/home/qhduan/Downloads/COCO/ve/local/lib/python2.7/site-packages/theano/sandbox/cuda/__init__.py:600: UserWarning: Your cuDNN version is more recent than the one Theano officially supports. If you see any problems, try updating Theano or downgrading cuDNN to version 5.
  warnings.warn(warn)
[nltk_data] Downloading package punkt to /home/qhduan/nltk_data...
[nltk_data]   Package punkt is already up-to-date!
train_size 82783
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 82783/82783 [00:08<00:00, 9803.66it/s]
vocabulary_size 8679
max_len 55
train_words_size 935568
(128, 4096) (128, 55) (128, 8679)
(128, 4096) (128, 55) (128, 8679)
(128, 4096) (128, 55) (128, 8679)
(128, 4096) (128, 55) (128, 8679)
(128, 4096) (128, 55) (128, 8679)
(128, 4096) (128, 55) (128, 8679)
(128, 4096) (128, 55) (128, 8679)
(128, 4096) (128, 55) (128, 8679)
(128, 4096) (128, 55) (128, 8679)
(128, 4096) (128, 55) (128, 8679)
(128, 4096) (128, 55) (128, 8679)
Epoch 1/20
935680/935680 [==============================] - 2629s - loss: 3.8930 - acc: 0.3291         
Epoch 2/20
935680/935680 [==============================] - 2692s - loss: 3.1218 - acc: 0.4042     
Epoch 3/20
935680/935680 [==============================] - 2672s - loss: 2.8787 - acc: 0.4265     
Epoch 4/20
935680/935680 [==============================] - 2565s - loss: 2.7405 - acc: 0.4391     
Epoch 5/20
935680/935680 [==============================] - 2529s - loss: 2.6423 - acc: 0.4487     
Epoch 6/20
935680/935680 [==============================] - 2529s - loss: 2.5686 - acc: 0.4566     
Epoch 7/20
935680/935680 [==============================] - 2529s - loss: 2.5025 - acc: 0.4631     
Epoch 8/20
935680/935680 [==============================] - 2549s - loss: 2.4513 - acc: 0.4685     
Epoch 9/20
935680/935680 [==============================] - 2707s - loss: 2.4130 - acc: 0.4728     
Epoch 10/20
935680/935680 [==============================] - 2700s - loss: 2.3732 - acc: 0.4774     
Epoch 11/20
935680/935680 [==============================] - 2721s - loss: 2.3421 - acc: 0.4817     
Epoch 12/20
935680/935680 [==============================] - 2726s - loss: 2.3110 - acc: 0.4857     
Epoch 13/20
935680/935680 [==============================] - 2692s - loss: 2.2815 - acc: 0.4894     
Epoch 14/20
935680/935680 [==============================] - 2671s - loss: 2.2587 - acc: 0.4918     
Epoch 15/20
935680/935680 [==============================] - 2755s - loss: 2.2357 - acc: 0.4952     
Epoch 16/20
935680/935680 [==============================] - 2605s - loss: 2.2185 - acc: 0.4976     
Epoch 17/20
935680/935680 [==============================] - 2629s - loss: 2.2002 - acc: 0.5000     
Epoch 18/20
935680/935680 [==============================] - 2652s - loss: 2.1852 - acc: 0.5021     
Epoch 19/20
935680/935680 [==============================] - 2640s - loss: 2.1760 - acc: 0.5039     
Epoch 20/20
935680/935680 [==============================] - 2619s - loss: 2.1618 - acc: 0.5055
