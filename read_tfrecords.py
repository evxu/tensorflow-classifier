import tensorflow as tf
import matplotlib.pyplot as plt
filename = 'validation.tfrecords'

def read_and_decode(filename):
	reader = tf.TFRecordReader()
	_,serialized_example = reader.read(filename)
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
    
	print 'height: ',height
	print 'width: ', width
    
	image_shape = tf.pack([height, width, 3])
	image = tf.reshape(image, image_shape)
	print 1, image.get_shape()
	images, labels = tf.train.shuffle_batch([image, label], shapes = [(299,299,3),()],
                                            batch_size = 2, capacity = 30, 
                                            num_threads=1, min_after_dequeue = 10)
	print 2, images.get_shape(), labels.get_shape()
	return images, labels



filename_queue = tf.train.string_input_producer([filename], num_epochs=10)
image, label = read_and_decode(filename_queue)
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in xrange(3):
        img, lab = sess.run([image, label])
        #print img[0,:,:,:].get_shape()
        print 'current batch'

        plt.imshow(img[0, :, :, :])
        plt.show()
        print lab[0]
        plt.imshow(img[1, :, :, :])
        plt.show()
        print lab[1]
    coord.request_stop()
    coord.join(threads)