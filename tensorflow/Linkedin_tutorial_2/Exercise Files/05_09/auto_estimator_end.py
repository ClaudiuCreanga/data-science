''' Classify MNIST images with a DNNClassifier '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Set flags
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
        'train_file', '../images/mnist_train.tfrecords', 'Training file')
tf.app.flags.DEFINE_string(
        'test_file', '../images/mnist_test.tfrecords', 'Test file')
tf.app.flags.DEFINE_string(
        'out_dir', 'dnn_output', 'Output directory')
tf.app.flags.DEFINE_integer(
        'train_steps', 8000, 'Number of training steps')
tf.app.flags.DEFINE_integer(
        'batch_size', 80, 'Number of images in each batch')
tf.app.flags.DEFINE_integer(
        'test_steps', 100, 'Number of tests to perform')

# Define constants
hidden_layers = [128, 32]
image_dim = 28
num_labels = 10

# Step 1: Parse MNIST data
def parser(record):
    features = tf.parse_single_example(record,
       features = {
       'images': tf.FixedLenFeature([], tf.string),
       'labels': tf.FixedLenFeature([], tf.int64),
       })
    image = tf.decode_raw(features['images'], tf.uint8)
    image.set_shape([image_dim * image_dim])
    image = tf.cast(image, tf.float32) * (1./255) - 0.5
    label = features['labels']
    return image, label

# Step 2: Create the feature column
column = tf.feature_column.numeric_column('pixels', 
                                          shape=[image_dim * image_dim])

# Step 3: Create the DNNClassifier
dnn_class = tf.estimator.DNNClassifier(hidden_layers, [column],
        model_dir=FLAGS.out_dir, n_classes=num_labels)

# Step 4: Train the estimator
def train_func():
    dataset = tf.data.TFRecordDataset(FLAGS.train_file)
    dataset = dataset.map(parser).repeat().batch(FLAGS.batch_size)
    image, label = dataset.make_one_shot_iterator().get_next()
    return {'pixels': image}, label

# Step 5: Test the estimator
def test_func():
    dataset = tf.data.TFRecordDataset(FLAGS.test_file)    
    dataset = dataset.map(parser).batch(FLAGS.batch_size)
    image, label = dataset.make_one_shot_iterator().get_next()
    return {'pixels': image}, label

# Create a TrainSpec
train_spec = tf.estimator.TrainSpec(train_func, max_steps=FLAGS.train_steps)


# Create an EvalSpec
eval_spec = tf.estimator.EvalSpec(test_func, steps=FLAGS.test_steps)

# Call train_and_evaluate
tf.estimator.train_and_evaluate(dnn_class, train_spec, eval_spec)