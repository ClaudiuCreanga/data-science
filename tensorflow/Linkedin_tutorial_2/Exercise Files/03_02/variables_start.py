''' Demonstrate the usage of tensors, graphs, and sessions '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Step 1: Create two variables and an operation that adds them together
t1 = tf.Variable(1)
t2 = tf.Variable(4)
t3 = t1 + t2


# Step 2: Obtain an operation that initializes the two variables
init = tf.global_variables_initializer()

# Step 3: Execute the initialization operation in a session
with tf.Session() as sess:
    sess.run(init)

# Step 4: Execute the addition operation in a session
    result = sess.run(t3)

# Step 5: Print the result of the addition to the log
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info("result {0}".format(result))