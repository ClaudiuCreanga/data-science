''' Classify MNIST images with a DNNClassifier '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Set flags

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
        model_dir='dnn_output', n_classes=num_labels)

# Step 4: Train the estimator
def train_func():
    dataset = tf.data.TFRecordDataset('../images/mnist_train.tfrecords')
    dataset = dataset.map(parser).repeat().batch(batch_size)
    image, label = dataset.make_one_shot_iterator().get_next()
    return {'pixels': image}, label
dnn_class.train(train_func, steps=num_steps)

# Step 5: Test the estimator
def test_func():
    dataset = tf.data.TFRecordDataset('../images/mnist_test.tfrecords')    
    dataset = dataset.map(parser).batch(batch_size)
    image, label = dataset.make_one_shot_iterator().get_next()
    return {'pixels': image}, label
metrics = dnn_class.evaluate(test_func)

# Create a TrainSpec

# Create an EvalSpec

# Call train_and_evaluate
