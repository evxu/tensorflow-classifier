# TensorFLow Classification Example
Simple tensorflow classification example codes. Works on small image datasets.

Supporting network achitectures includes standard networks under [tensorflow.contrib.slim.nets](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim/python/slim/nets)
        
        @@alexnet_v2
        @@inception_v1
        @@inception_v1_base
        @@inception_v2
        @@inception_v2_base
        @@inception_v3
        @@inception_v3_base
        @@overfeat
        @@vgg_a
        @@vgg_16


## TFRecords: data are pre-processed into [TFRecords](https://www.tensorflow.org/programmers_guide/reading_data) format

 * convert_to_records.py: create TFRecords data file from raw images
 
 * example: DHL logo apperance binary classification
    training dataset: around 800 images
    validation dataset: around 200 images

 * train.py train a VGG network

 * test.py restore checkpoint and test network on validation dataset

 * train2.py train and do validation periodically.


## Picpac: data are pre-processed into [picpac](http://picpac.readthedocs.io/en/latest/) format
 * train_picpac.py

## Docker image
Networks can be trained on GPU with tensorflow installed, or in a docker container on CPU.
To build this docker image, use Dockerfile provided in tensorflow-docker repository.

`$ docker build -f /path/to/a/Dockerfile .`
