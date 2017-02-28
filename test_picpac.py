from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import time
import argparse
import sys
import tensorflow.contrib.slim.nets as net
import os
from train_picpac import inference
import math
import numpy as np
import cv2
import picpac
import pkgutil

# referece: https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10_eval.py


def eval():
	picpac_config = dict(seed=2016,
				shuffle=False,
				reshuffle=False,
				# max_size = 400,
				# resize_width=FLAGS.img_size,
				# resize_height=FLAGS.img_size,
				batch=1,
				pert_angle=5,
				pert_hflip=True,
				pert_vflip=False,
				channels=3,
				channel_first=False # this is tensorflow specific
								# Caffe's dimension order is different.
				)
	te_stream = picpac.ImageStream(FLAGS.test_db, perturb=False, loop=False, **picpac_config)
	with tf.Graph().as_default():
		images_placeholder = tf.placeholder(tf.float32, shape = (1, None, None, 3), name = 'images')
		labels_placeholder = tf.placeholder(tf.int32, shape = (1,), name = 'labels')
		
		resized_images = tf.image.resize_images(images_placeholder, size=(FLAGS.img_size, FLAGS.img_size))
		# inference model
		logits, _ = inference(resized_images, FLAGS.num_classes, FLAGS.net)
		scores = tf.nn.softmax(logits, dim=-1, name=None)
		# Calculate predictions
		top_k_op = tf.nn.in_top_k(predictions=logits, targets=labels_placeholder, k=FLAGS.num_classes-1, name="accuracy")

		checkpoint_file = os.path.join(FLAGS.checkpoint_dir,'model.ckpt-' + FLAGS.ckpt_step)
		saver = tf.train.Saver()

		with tf.Session() as sess:
			saver.restore(sess, checkpoint_file)
			htmlf = open(os.path.join(FLAGS.outdir, FLAGS.html), 'w')
			num = 0
			true_count = 0
			for images, labels, pad in te_stream:
				
				predictions, correct = sess.run([scores, top_k_op], feed_dict={images_placeholder: images, labels_placeholder: labels})
				print('score: %s, correct: %s'%(predictions, np.sum(correct)))
				# save images
				path = os.path.join(FLAGS.outdir, FLAGS.imagedir, str(num)+'.jpg')
				cv2.imwrite(path, images[0])
				# wirte to html
				htmlf.write("<div style='float:left; width:400;height:350;'><img src={} width='300'><br>score={}<br>groundtruth={}<br>prediction={}</div>\n".format(
					os.path.join(FLAGS.imagedir, str(num)+'.jpg'), predictions[0][1], labels[0], correct[0]))

				num += 1
				true_count += np.sum(correct)
			# Compute precisions
			precision = true_count / num
			print('precision = %.4f' % (precision))
			
			htmlf.write('<br>total accuracy = {}\n'.format(precision))
			htmlf.close()
			

def main(_):
	start_time = time.time()
	try:
		os.mkdir(os.path.join(FLAGS.outdir, FLAGS.imagedir))
	except:
		pass

	eval()
	print('evaluation took %.2f sec' % (time.time()-start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
    parser.add_argument('--img_size', type=int, default=224, help='Image witdh and height')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--test_db', type=str, default=None, help='picpac test db')
    #parser.add_argument('--max_steps', type=int, default=10000, help='Number of steps to run trainer')
    parser.add_argument('--checkpoint_dir', type=str, default='./log/run1/', help='Directory where to read model checkpoints')
    #parser.add_argument('--num_examples', type=int, default=200, help='Number of examples to run')
    parser.add_argument('--ckpt_step', type=str, default = '8999', help='Checkpoint step to restore')
    parser.add_argument('--outdir', type=str, default = 'vals', help='dir to store test images')
    parser.add_argument('--imagedir', type=str, default = 'images', help='dir to store test images')
    parser.add_argument('--html', type=str, default = 'val.html', help='html page to write test results')
    parser.add_argument('--net', type=str, default='vgg.vgg_16', help='cnn architecture')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run()