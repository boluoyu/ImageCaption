
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

