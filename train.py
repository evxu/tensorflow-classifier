from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import time
import argparse
import sys
# import pkgutil
import tensorflow.contrib.slim.nets as net
import os

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
    resized_image = tf.image.resize_image_with_crop_or_pad(image, img_size, img_size)
    # not work: resized_image = tf.image.resize_images(resized_image, [SIZE, SIZE],align_corners=True)
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
#     full = 'tensorflow.contrib.slim.python.slim.nets.' + 'vgg.vgg_16'
#     # e.g. full == 'tensorflow.contrib.slim.python.slim.nets.vgg.vgg_a'
#     fs = full.split('.')
#     loader = pkgutil.find_loader('.'.join(fs[:-1]))
#     module = loader.load_module('')
#     net = getattr(module, fs[-1])
    return net.vgg.vgg_16(inputs, num_classes)

def fcn_loss (logits, labels):
    with tf.name_scope('loss'):
        labels = tf.to_int32(labels)    # float from picpac
        xe = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='xentropy')
        hit = tf.cast(tf.nn.in_top_k(logits, labels, 1, name="accuracy"), tf.float32)
        return [tf.reduce_mean(xe, name='xentropy_mean'), tf.reduce_mean(hit, name='accuracy_total')]
    pass

def do_eval(sess, dataset):
    # run evaluation against the full epoch of data

    pass

def run_training(start_time):
    # use VGG model
    slim = tf.contrib.slim
    # Tell TensorFlow that the model will be built into the default Graph.
    with tf.Graph().as_default():
        # Input images and labels.
        images_placeholder = tf.placeholder(tf.float32, shape = (FLAGS.batch_size, FLAGS.img_size, FLAGS.img_size, 3), name = 'images')
        labels_placeholder = tf.placeholder(tf.float32, shape = (FLAGS.batch_size,), name = 'labels')
        
        images, labels = inputs(FLAGS.train_data, FLAGS.img_size, FLAGS.batch_size)

        images_placeholder = images
        labels_placeholder = labels
        tf. summary.image('input', images_placeholder, 10)

        predictions, _ = inference(images_placeholder, FLAGS.num_classes)
        # Specify the loss function:
        # loss = slim.losses.softmax_cross_entropy(predictions, labels)
        loss, accuracy = fcn_loss(predictions, labels_placeholder)
        tf.summary.scalar('xentropy_mean', loss)
        tf.summary.scalar('accuracy_mean', accuracy)

        # Specify the optimization scheme:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        # train_op = slim.learning.create_train_op(loss, optimizer)
        train_op = optimizer.minimize(loss)
        tf.summary.scalar('learning_rate', FLAGS.learning_rate)
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
        config.gpu_options.allow_growth = True
        
        loss_sum = 0
        acc_sum = 0
        batch_sum =0

        with tf.Session(config=config) as sess:
            sess.run(init_op)
            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            for step in xrange(FLAGS.max_steps):
                _, loss_value, acc_value = sess.run([train_op, loss, accuracy])
                loss_sum += loss_value * FLAGS.batch_size
                acc_sum += acc_value * FLAGS.batch_size
                batch_sum += FLAGS.batch_size
                # Print an overview fairly often.
                if step % 100 == 0:
                    duration = time.time() - start_time
                    print('Step %d: loss = %.4f, accuracy = %.4f (%.3f sec)' % 
                        (step, loss_sum/batch_sum, acc_sum/batch_sum, duration))

                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                    loss_sum = 0
                    acc_sum = 0
                    batch_sum =0
                    start_time = time.time()

                # Save a checkpoint and evaluate the model periodically
                if (step + 1)%1000==0 or (step+1)==FLAGS.max_steps:
                    checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_file, global_step = step)
                    # Evaluate against the training set
                    pass

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
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--img_size', type=int, default=244, help='Image witdh and height')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--train_data', type=str, default='train.tfrecords', help='TFRecords filename of training data')
    parser.add_argument('--eval_data', type=str, default='validation.tfrecords', help='TFRecords filename of validation data')
    parser.add_argument('--log_dir', type=str, default='./temp_log', help='Directory to put the log data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run()