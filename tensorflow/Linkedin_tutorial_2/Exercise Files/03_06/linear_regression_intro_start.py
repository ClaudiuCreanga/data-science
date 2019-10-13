''' Demonstrates linear regression with TensorFlow '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Set constants
N = 1000
learning_rate = 0.1
batch_size = 40  # the size of the part of the entire dataset, we don't feed the entire dataset all at once, we divide it into parts
num_batches = 400
# Step 1: Generate input points

x = np.random.normal(size=N)
m_real = np.random.normal(loc=0.5, scale=0.2, size= N)
b_real = np.random.normal(loc=1.0, scale=0.2, size= N)
y = m_real*x + b_real

# Step 2: Create variables and placeholders
m= tf.Variable(tf.random_normal([]))
b= tf.Variable(tf.random_normal([]))

x_holder = tf.placeholder(tf.float32, shape=[batch_size])
y_holder = tf.placeholder(tf.float32, shape=[batch_size])
# Step 3: Define model and loss
model = m * x_holder + b
loss = tf.reduce_mean(tf.pow(model - y_holder, 2))
# Step 4: Create optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# Step 5: Execute optimizer in a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for _ in range(num_batches):
        x_data = np.empty(batch_size)
        y_data = np.empty(batch_size)

        for i in range(batch_size):
            index = np.random.randint(0, N)
            x_data[i] = x[index]
            y_data[i] = y[index]

        sess.run(optimizer, feed_dict={x_holder: x_data, y_holder: y_data})

    print("m = ", sess.run(m))
    print("b = ", sess.run(b))