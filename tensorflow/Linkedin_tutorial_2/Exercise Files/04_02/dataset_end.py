''' Demonstrates the usage of datasets and iterators '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# Generator function
def gen_func():
    x = 12
    while x < 20:
        yield x
        x += 2

# Step 1: Create a dataset/iterator from a range of values
ds1 = tf.data.Dataset.range(4)
iter1 = ds1.make_one_shot_iterator()

# Step 2: Create a dataset/iterator from two tensors
t1 = tf.constant([4, 5])
t2 = tf.constant([6, 7])
ds2 = tf.data.Dataset.from_tensors([t1, t2])
iter2 = ds2.make_one_shot_iterator()

# Step 3: Create a dataset/iterator from rows of a tensor
t3 = tf.constant([[8], [9], [10], [11]])
ds3 = tf.data.Dataset.from_tensor_slices(t3)
iter3 = ds3.make_one_shot_iterator()

# Step 4: Create a dataset/iterator from a generator function
ds4 = tf.data.Dataset.from_generator(gen_func, output_types=tf.int64)
iter4 = ds4.make_one_shot_iterator()

# Step 5: Print the elements of each dataset
with tf.Session() as sess:
    
    for _ in range(4):
        print(sess.run(iter1.get_next()))
        
    print(sess.run(iter2.get_next()))        

    for _ in range(4):
        print(sess.run(iter3.get_next()))    
        
    for _ in range(4):
        print(sess.run(iter4.get_next()))      