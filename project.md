# Definition

## Project Overview

> Student provides a high-level overview of the project in layman’s terms. Background
> information such as the problem domain, the project origin, and related data sets or input
> data is given.

This project present a model that generate natural language description of images.
I built a web app which user could upload an image, and generate a description.

This application uses MS-COCO datasets to train and test.

## Problem Statement

> The problem which needs to be solved is clearly defined. A strategy for solving the problem,
> including discussion of the expected solution, has been made.

The final results of this project is description of images.

First of all, I download MS-COCO dataset.
It includes a lot of images and their captions.

Secondly, I will build a deep learning model with Convolutional Neural Network and
Long-short Term Memory network to generate the description.

Finally, build a webapp with python and a webserver develop kit named flask.
Let user could upload a image and get the description.

## Metrics

> Metrics used to measure performance of a model or result are clearly defined. Metrics are
> justified based on the characteristics of the problem.

I will use BLEU to measure the performance of model.

BLEU (bilingual evaluation understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. Quality is considered to be the correspondence between a machine's output and that of a human.

BLEU is a useful measurement of machine translate, which is from one language (like English) to another language (maybe French).
This project is kindof like a machine tranlate quest, which if from one image, to language (English sentence).

# Analysis

## Data Exploration

> If a dataset is present, features and calculated statistics relevant to the problem have been
> reported and discussed, along with a sampling of the data. In lieu of a dataset, a thorough
> description of the input space or input data has been made. Abnormalities or
> characteristics about the data or input that need to be addressed have been identified.

The COCO (Common Objects in Context) dataset has a lot of annotated images,
you can download from its [website](http://mscoco.org/).
COCO is a image recognition, segmentation, and captioning dataset.

I'm only using the captioning data as target.  
In 2014 training data, there are 82783 images with caption.  
In 2014 validation data, there are 40504 images with caption.

Each image is colored image and in JPEG format.  
Each image has different content, such as people, mobile, animal, and so on.

Each image has a description, for example:

Fig. 1 COCO image: A bunch of trays that have different food.

![A bunch of trays that have different food.](coco-01.png)

Fig. 2 COCO image: A giraffe standing next to a forest filled with trees.

![A giraffe standing next to a forest filled with trees.](coco-02.png)

## Exploratory Visualization

> A visualization has been provided that summarizes or extracts a relevant characteristic or
> feature about the dataset or input data with thorough discussion. Visual cues are clearly
> defined.

Each description of images has 6 to 55 words. Most of sentences have 9 to 12 words.

Fig. 3 A plot showing words length histogram.

![words-length.png](words-length.png)

Fig. 4 Most words in image captions.

Most of words are adjective, like: colored, red, high, blue. Also a lot of noun, kite, building, dog, bed and so on.

![words-cloud.png](words-cloud.png)

## Algorithms and Techniques

> Algorithms and techniques used in the project are thoroughly discussed and properly justified based on the characteristics of the problem.
> Benchmark Student clearly defines a benchmark result or threshold for comparing performances of
> solutions obtained.


By using Convolutional Neural Networks (CNN) over images,
Long-short Term Memory (LSTM) network to read partial human-language.
Finally, another LSTM to generate the description of images.

I use flask, a light-weight python web server kit to build a web server.

Algorithm needs to read the image and generate description based image.
This model use a CNN algorithm with VGG-16 pretrain weights to read image.
Other input is a LSTM network, trying to read partial natural language.
The output of model is one word of sentence.

For example, we have a image of flower, and I want the model to descript the image "this is flower".
First of all, I feed the model with the image and a tag of `<start>`, and I hope the model will give us `this`.
Secondly, I feed the model with the image and partial sentence `<start> this`, The model will give us `is`.

Image and (<start>) -> (this)  
Image and (<start> this) -> (is)  
Image and (<start> this is) -> (flower)  
Image and (<start> this is flower) -> (<end>)

When model give us `<end>`, I will know it finish a sentence.

# Methodology

## Data Preprocessing

> All preprocessing steps have been clearly documented. Abnormalities or characteristics
about the data or input that needed to be addressed have been corrected. If no data
preprocessing is necessary, it has been clearly justified.

For each image, I convert it to size of 224x224. I use a python module named pillow to do this work.

Then I use VGG-16 model to convert it into a 4096 vector. I use a deep learning software keras, load a pretrained VGG-16 model, convert the image and save the vectors.

## Implementation

> The process for which metrics, algorithms, and techniques were implemented with the
given datasets or input data has been thoroughly documented. Complications that occurred
during the coding process are discussed.

There're three step in this projects.

* Data processing. Resize image to 224x224 and use VGG-16 model vectorize them.
* Training model, using image vector and image caption.
* Develop webapp.

Fig. 5 The computational graph, includes the network architecture.

![image-caption.png](image-caption.png)

## Refinement

> The process of improving upon the algorithms and techniques used is clearly documented.
Both the initial and final solutions are reported, along with intermediate solutions, if
necessary.

## Results

## Model Evaluation and Validation

> The final model’s qualities — such as parameters — are evaluated in detail. Some type of analysis is used to validate the robustness of the model’s solution.

## Justification

> The final results are compared to the benchmark result or threshold with some type of
statistical analysis.Justification is made as to whether the final model and solution is
significant enough to have adequately solved the problem.

# Conclusion

## Free-Form Visualization

> A visualization has been provided that emphasizes an important quality about the project
with thorough discussion. Visual cues are clearly defined.

## Reflection

> Student adequately summarizes the end-to-end problem solution and discusses one or two
particular aspects of the project they found interesting or difficult.



## Improvement
> Discussion is made as to how one aspect of the implementation could be improved.
Potential solutions resulting from these improvements are considered and
compared/contrasted to the current solution.

# Quality

## Presentation

> Project report follows a well-organized structure and would be readily understood by its
intended audience. Each section is written in a clear, concise and specific manner. Few
grammatical and spelling mistakes are present. All resources used to complete the project
are cited and referenced.

## Functionality

> Code is formatted neatly with comments that effectively explain complex implementations.
Output produces similar results and solutions as to those discussed in the project.
