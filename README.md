# TensorFLow Classification Example
Simple tensorflow classification example codes. Works on small image datasets.

Supporting network achitectures includes networks under tensorflow.contrib.slim.nets


#### TFRecords version: data are pre-processed into [TFRecords](https://www.tensorflow.org/programmers_guide/reading_data) format

 * example: DHL logo binary classification
    training dataset: around 800 images
    validation dataset: around 200 images

 * train.py train a VGG network

 * test.py restore checkpoint and test network on validation dataset

 * train2.py train and do validation periodically.

=====
#### Picpac version: data are pre-processed into [picpac](http://picpac.readthedocs.io/en/latest/) format
 * rewrite train2.py to train with picpac format dataset
