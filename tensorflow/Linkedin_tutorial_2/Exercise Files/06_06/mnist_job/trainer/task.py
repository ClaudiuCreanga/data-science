'''Execute a TensorFlow experiment in the cloud'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import tensorflow as tf

# Set parameters
image_dim = 28
hidden_layers = [128, 32]
num_labels = 10

# Function to parse TFRecords
def tf_parser(record):
    features = tf.parse_single_example(record,
        features={
          'images': tf.FixedLenFeature([], tf.string),
          'labels': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['images'], tf.uint8)
    image.set_shape([image_dim * image_dim])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = features['labels']
    return image, label

# Return train function
def create_train_func(data_dir, batch_size):
    train_file = os.path.join(data_dir, 'mnist_train.tfrecords')
    def train_func():
        dataset = tf.data.TFRecordDataset(train_file)
        dataset = dataset.map(tf_parser).repeat().batch(batch_size)
        image, label = dataset.make_one_shot_iterator().get_next()
        return {'pixels': image}, label
    return train_func

# Return test function
def create_test_func(data_dir, batch_size):
    test_file = os.path.join(data_dir, 'mnist_test.tfrecords')
    def test_func():
        dataset = tf.data.TFRecordDataset(test_file)
        dataset = dataset.map(tf_parser).batch(batch_size)
        image, label = dataset.make_one_shot_iterator().get_next()
        return {'pixels': image}, label
    return test_func

# Process arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Training arguments
    parser.add_argument(
        '--data-dir',
        help='Directory/bucket containing training/test data',
        required=True
    )
    
    # Batch size for training
    parser.add_argument(
        '--train_batch_size',
        help='Batch size for training steps',
        type=int,
        default=80
    )

    # Number of training steps
    parser.add_argument(
        '--train_steps',
        help='Number of training steps per epoch',
        type=int,
        default=8000
    )    
    
    # Output arguments
    parser.add_argument(
        '--job-dir',
        help='Directory/bucket for storing output',
        required=True
    )

    # Batch size for testing
    parser.add_argument(
        '--test_batch_size',
        help='Batch size for testing',
        type=int,
        default=80
    )
    
    # Number of testing steps
    parser.add_argument(
        '--test-steps',
        help='Number of testing steps',
        default=100,
        type=int
    )    
    
    args = parser.parse_args()
    arguments = args.__dict__

    # Create the estimator
    column = tf.feature_column.numeric_column('pixels',
        shape=[image_dim * image_dim])
    dnn_class = tf.estimator.DNNClassifier(hidden_layers, [column],
        n_classes=num_labels, model_dir=arguments['job_dir'])

    # Create the train/test functions
    train_func = create_train_func(arguments['data_dir'], arguments['train_batch_size'])
    test_func = create_test_func(arguments['data_dir'], arguments['test_batch_size'])

    # Train and evaluate the estimator
    train_spec = tf.estimator.TrainSpec(train_func, max_steps=arguments['train_steps'])
    eval_spec = tf.estimator.EvalSpec(test_func, steps=arguments['test_steps'])
    tf.estimator.train_and_evaluate(dnn_class, train_spec, eval_spec)



