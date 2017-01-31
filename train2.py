from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import time
import argparse
import sys
import pkgutil
import tensorflow.contrib.slim.nets as net
import os
import math
import numpy as np 

'''
Train and validation version 2
Difference training and validation by changing the feeding data 
Do validation over num_example/batch_size steps.

Following FLAG.net have worked for samll datasets:
    vgg.vgg_16  img_size=244
    vgg.vgg_a   img_size=244
    inception_v3.inception_v3   img_size=299
    alexnet.alexnet_v2  img_size=224

Failed to converge after 10000 stpes:
    inception_v1.inception_v1   img_size=224

'''

def read_and_decode(filename_queue, img_size):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, 
        features = {
        'height': tf.FixedLenFeature([],tf.int64), 
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image_shape = tf.pack([height, width, 3])
    image = tf.reshape(image, image_shape)
    # resize image for the nets
    # crop or pad resize: resized_image = tf.image.resize_image_with_crop_or_pad(image, img_size, img_size)
    resized_image = tf.image.resize_images(image, [img_size, img_size])
    # initiralize image values
    resized_image = tf.cast(resized_image, tf.float32) * (1. / 255) - 0.5
    return resized_image, label

def inputs(dataset, img_size, batch_size):
    filename = dataset
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=None)
        image, label = read_and_decode(filename_queue, img_size)
        images, labels = tf.train.shuffle_batch([image, label], shapes = [(img_size, img_size,3),()],
                                                batch_size = batch_size, capacity = 300, 
                                                num_threads=10, 
                                                min_after_dequeue = 1)
        
        return images, labels

def inference (inputs, num_classes):
    full = 'tensorflow.contrib.slim.python.slim.nets.' + FLAGS.net
    # e.g. full == 'tensorflow.contrib.slim.python.slim.nets.vgg.vgg_16'
    fs = full.split('.')
    loader = pkgutil.find_loader('.'.join(fs[:-1]))
    module = loader.load_module('')
    net = getattr(module, fs[-1])
    # return net.vgg.vgg_16(inputs, num_classes)
    return net(inputs, num_classes)

def fcn_loss (logits, labels):
    with tf.name_scope('loss'):
        labels = tf.to_int32(labels)    # float from picpac
        xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
        hit = tf.cast(tf.nn.in_top_k(logits, labels, 1, name="accuracy"), tf.float32)
        return [tf.reduce_mean(xe, name='xentropy_mean'), tf.reduce_mean(hit, name='accuracy_total')]
    pass


def do_eval(sess, top_k_op, images, labels, images_placeholder, labels_placeholder, num_examples):
    """
    compute the average accuracy on num_examples examples with current defualt graph and its variables
    this can be used for validaton, and can also be used to go through training data.
    by passing corresponding images and label tensors.
    """
    # Compute number of steps in an epoch
    num_iter = int(math.ceil(num_examples / FLAGS.batch_size))
    true_count = 0 # Counts the number of correct predictions.
    total_sample_count = num_iter * FLAGS.batch_size

    eval_step = 0
    count = 0
    while eval_step < num_iter: # and not eval_coord.should_stop():
        # get feeding data
        x, y = sess.run([images, labels])
        feed_dict = {images_placeholder: x,
                labels_placeholder: y}
        predictions = sess.run(top_k_op, feed_dict = feed_dict)
        true_count += np.sum(predictions)
        eval_step += 1
    # Compute precisions
    precision = true_count / total_sample_count
    return precision 


def run_training(start_time):
    # use VGG model
    slim = tf.contrib.slim
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Input images and labels.
        images_placeholder = tf.placeholder(tf.float32, shape = (FLAGS.batch_size, FLAGS.img_size, FLAGS.img_size, 3), name = 'images')
        labels_placeholder = tf.placeholder(tf.float32, shape = (FLAGS.batch_size,), name = 'labels')
        
        # training data queue
        tr_images, tr_labels = inputs(FLAGS.train_data, FLAGS.img_size, FLAGS.batch_size)
        tf. summary.image('train_input', tr_images, 20)
        
        # validation data queue
        eval_images, eval_labels = inputs(FLAGS.eval_data, FLAGS.img_size, FLAGS.batch_size)
        tf.summary.image('eval_input', eval_images, 20)
        
        # model inference, loss and accuracy
        predictions, _ = inference(images_placeholder, FLAGS.num_classes)
        loss, accuracy = fcn_loss(predictions, labels_placeholder)
        
        # Prediction results in a batch.
        top_k_op = tf.nn.in_top_k(predictions, tf.to_int32(labels_placeholder), 1, name="accuracy")

        tf.summary.scalar('xentropy_mean', loss)
        tf.summary.scalar('accuracy_mean', accuracy)

        # Specify the optimization scheme:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        train_op = optimizer.minimize(loss)
        tf.summary.scalar('learning_rate', FLAGS.learning_rate)
        tf.summary.scalar('batch_size', FLAGS.batch_size)


        # create two variables to summarize evaluation results
        # initial evaluation accuracy to 0.5
        eval_tr = tf.Variable(0.5, name="eval_validationdata")
        eval_te = tf.Variable(0.5, name="eval_trainingdata")
        eval_s1 = tf.summary.scalar('eval_validationdata', eval_te)
        eval_s2 = tf.summary.scalar('eval_trainingdata', eval_tr)

        # The op for initializing the variables.
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(max_to_keep = 10)
        # Build the summary Tensor based on the TF collection of Summaries 
        summary_op = tf.summary.merge_all()
        # Instantiate a SummaryWriter to output summaries and the Graph
        summary_writer = tf.summary.FileWriter(FLAGS.log_dir, tf.get_default_graph())

        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        
        loss_sum = 0
        acc_sum = 0
        batch_sum =0

        with tf.Session(config=config) as sess:
            sess.run(init_op)
            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            for step in xrange(FLAGS.max_steps+1):
                # Get value of a batch of training images and labels
                x, y = sess.run([tr_images, tr_labels])
                feed_dict = {images_placeholder: x,
                            labels_placeholder: y}
                # start training
                _, loss_value, acc_value = sess.run([train_op, loss, accuracy], feed_dict = feed_dict)
                loss_sum += loss_value * FLAGS.batch_size
                acc_sum += acc_value * FLAGS.batch_size
                batch_sum += FLAGS.batch_size
                # Print an overview fairly often.
                if step % 100 == 0:
                    duration = time.time() - start_time
                    print('Step %d: loss = %.4f, accuracy = %.4f (%.3f sec)' % 
                        (step, loss_sum/batch_sum, acc_sum/batch_sum, duration))

                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                    loss_sum = 0
                    acc_sum = 0
                    batch_sum =0

                # Save a checkpoint and evaluate the model periodically
                if step!=0:
                    if step%1000 == 0 or step == FLAGS.max_steps:
                        checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_file, global_step = step)
                        
                        # Evaluate against the training set
                        tr_precision = do_eval(sess, top_k_op, tr_images, tr_labels, images_placeholder, labels_placeholder, FLAGS.num_tr_examples)
                        print('Step %d: training data set evaluation: %.4f (%.f sec)' %
                            (step, tr_precision, time.time()-start_time))

                        # Evaluate against the validation set
                        val_precision = do_eval(sess, top_k_op, eval_images, eval_labels, images_placeholder, labels_placeholder, FLAGS.num_val_examples)
                        print('Step %d: validation data set evaluation: %.4f (%.f sec)' %
                            (step, val_precision, time.time()-start_time))
                        
                        # update validation results
                        sess.run(eval_tr.assign(tr_precision))
                        sess.run(eval_te.assign(val_precision))
                        # print(eval_tr.eval())
                        # print(eval_te.eval())
                        merged_str = sess.run(tf.summary.merge([eval_s1, eval_s2]), feed_dict=feed_dict)
                        summary_writer.add_summary(merged_str, step)
                        summary_writer.flush()

            print('Done training for %d steps.' % (FLAGS.max_steps))
            # When done, ask the threads to stop.
            coord.request_stop()
            # Wait for threads to finish.
            coord.join(threads)


def main(_):
    start_time = time.time()
    run_training(start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate')
    # parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs to run trainer')
    parser.add_argument('--max_steps', type=int, default=10000, help='Number of steps to run trainer')
    parser.add_argument('--batch_size', type=int, default=30, help='Batch size')
    parser.add_argument('--img_size', type=int, default=224, help='Image witdh and height')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--train_data', type=str, default='train.tfrecords', help='TFRecords filename of training data')
    parser.add_argument('--eval_data', type=str, default='validation.tfrecords', help='TFRecords filename of validation data')
    parser.add_argument('--log_dir', type=str, default='./temp_log/', help='Directory to put the log data')
    parser.add_argument('--num_val_examples', type=int, default=200, help='Number of validation examples to run')
    parser.add_argument('--num_tr_examples', type=int, default=800, help='Number of training examples to run')
    parser.add_argument('--net', type=str, default='vgg.vgg_16', help='cnn architecture' )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run()