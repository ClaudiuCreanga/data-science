''' Demonstrate how optimizers are used '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Define constants
learn_rate = 0.2
num_steps = 100

# Step 1: Define the loss function
x = tf.Variable(0.0)
loss = tf.pow(x, 2) - 4.0 * x + 5.0

# Step 2: Create an optimizer to minimize the loss
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(loss)

# Step 3: Execute the optimizer in a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(num_steps):
        _, loss_val, x_val = sess.run([optimizer, loss, x])

# Step 4: Print the values of x and the loss to the log
    tf.logging.set_verbosity(tf.logging.INFO)
    str = 'x is {0} and the loss is {1}'.format(x_val, loss_val)
    tf.logging.info(str)