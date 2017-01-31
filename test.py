from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import time
import argparse
import sys
import tensorflow.contrib.slim.nets as net
import os
from train import inputs, inference
import math
import numpy as np

# referece: https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10_eval.py
def eval():
	with tf.Graph().as_default():

		images, labels = inputs(FLAGS.eval_data, FLAGS.img_size, FLAGS.batch_size)
		tf.summary.image('eval_input', images, 20)
		# inference model
		logits, _ = inference(images, FLAGS.num_classes)
		# Calculate predictions
		top_k_op = tf.nn.in_top_k(logits, labels, 1, name="accuracy")

		# Restore the moving average version of the learned variables for eval.
		# ema = tf.train.ExponentialMovingAverage(decay=0.9999)
		# saver = tf.train.Saver(ema.variables_to_restore)

		# summary_op = tf.merge_all_summaries()
		# graph_def = tf.get_default_graph().as_graph_def()
		# summary_writer = tf.train.SummaryWriter(FLAGS.log_dir, graph_def=graph_def)
		checkpoint_file = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-' + FLAGS.ckpt_step)
		saver = tf.train.Saver()

		with tf.Session() as sess:
			# ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
			# if ckpt and ckpt.model_checkpoint_path:
			# 	# restore from checkpoint
			# 	saver.restore(sess, checkpoint_file)
			# 	global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
			# 	print ('global_step = %d' % global_step)
			# else:
			# 	print('No checkpoint file found')
			# 	return
			saver.restore(sess, checkpoint_file)

			# Start the queue runners.
			coord = tf.train.Coordinator()
			threads = tf.train.start_queue_runners(coord=coord)
			num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
			true_count = 0 # Counts the number of correct predictions.
			total_sample_count = num_iter * FLAGS.batch_size
			step = 0
			while step < num_iter and not coord.should_stop():
				predictions = sess.run([top_k_op])
				true_count += np.sum(predictions)
				print(step, true_count)
				step += 1

			# Compute precisions
			precision = true_count / total_sample_count
			print('precision = %.4f' % (precision))

			coord.request_stop()
			coord.join(threads)

def main(_):
	start_time = time.time()
	eval()
	print('evaluation took %.2f sec' % (time.time()-start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--img_size', type=int, default=244, help='Image witdh and height')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--train_data', type=str, default='train.tfrecords', help='TFRecords filename of training data')
    parser.add_argument('--eval_data', type=str, default='validation.tfrecords', help='TFRecords filename of validation data')
    #parser.add_argument('--max_steps', type=int, default=10000, help='Number of steps to run trainer')
    parser.add_argument('--checkpoint_dir', type=str, default='./log/run1/', help='Directory where to read model checkpoints')
    parser.add_argument('--num_examples', type=int, default=200, help='Number of examples to run')
    parser.add_argument('--log_dir', type=str, default='./log/train1/', help='Directory to put the log data')
    parser.add_argument('--ckpt_step', type=str, default = '8999', help='Checkpoint step to restore')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run()