''' Demonstrates linear regression with TensorFlow '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Set constants
N = 1000
learn_rate = 0.1
batch_size = 40
num_batches = 400

# Step 1: Generate input points
x = np.random.normal(size=N)
m_real = np.random.normal(loc=0.5, scale=0.2, size=N)
b_real = np.random.normal(loc=1.0, scale=0.2, size=N)
y = m_real * x + b_real

# Step 2: Create variables and placeholders
m = tf.Variable(tf.random_normal([]))
b = tf.Variable(tf.random_normal([]))
gstep = tf.Variable(0, trainable=False)
x_holder = tf.placeholder(tf.float32, shape=[batch_size])
y_holder = tf.placeholder(tf.float32, shape=[batch_size])

# Step 3: Define model and loss
model = m * x_holder + b
loss = tf.reduce_mean(tf.pow(model - y_holder, 2))

# Step 4: Create optimizer
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, 
                                             global_step=gstep)

op1 = tf.summary.scalar('m', m)
op2 = tf.summary.scalar('b', b)
merged_op = tf.summary.merge_all()

file_writer = tf.summary.FileWriter('tboard')


# Step 5: Execute optimizer in a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Perform training
    for batch in range(num_batches):

        # Create batch of data
        x_data = np.empty(batch_size)
        y_data = np.empty(batch_size)
        for i in range(batch_size):
            index = np.random.randint(0, N)
            x_data[i] = x[index]
            y_data[i] = y[index]

        _, summary, step = sess.run([optimizer, merged_op, gstep],
                    feed_dict={x_holder: x_data, y_holder: y_data})
        
        file_writer.add_summary(summary, global_step=step)
        file_writer.flush()