from PIL import Image
import numpy as np
import tensorflow as tf
import os
import argparse

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(data_dir, tfrecords_filename):
    labels = os.listdir(data_dir)
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    for label in labels:
        path = os.path.join(data_dir, label)
        img_files = os.listdir(path)
        for file in img_files:
            # The reason to store image sizes was demonstrated
            # in the previous example -- we have to know sizes
            # of images to later read raw serialized string,
            # convert to 1d array and convert to respective
            # shape that image used to have.
            img_path = os.path.join(path, file)
            img = np.array(Image.open(img_path))
            height = img.shape[0]
            width = img.shape[1]
            img_raw = img.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'label': _int64_feature(int(label))}))
            writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./tempimg', help='original data folder')
    parser.add_argument('--TFRfilename', type=str, default='img.tfrecords', help='TFRecords filename')
    args = parser.parse_args()
    main(data_dir = args.data_dir, tfrecords_filename = args.TFRfilename)