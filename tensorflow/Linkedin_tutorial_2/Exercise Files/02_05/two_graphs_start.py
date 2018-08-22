''' Demonstrate the usage of tensors, graphs, and sessions '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Step 1: Create two tensors and an addition operation
t1 = tf.constant([1, 4, 5.6])
t2 = tf.random_normal([3])
t3 = t1 + t2

graph1 = tf.get_default_graph()
# Step 2: Create a second graph and make it the default graph
graph2 = tf.Graph()
with graph2.as_default():

# Step 3: Create two tensors in the second graph and a subtraction operation
    t4 = tf.constant([3,6,5.5])
    t5 = tf.random_normal([3])
    t6 = t4 - t5

# Step 4: Create a session and execute the addition operation from the first graph
with tf.Session(graph = graph1) as sess:
    print("Addition", sess.run(t3))

# Step 5: Create a second session and execute the subtraction operation from the second graph
with tf.Session(graph = graph2) as sess:
    print("Subs", sess.run(t6))
