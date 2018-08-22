import tensorflow as tf
import numpy as np

# hyper parameters
N = 1000
batch_size = 50
num_batches = 500
learning_rate = 0.1

# generate the input points
x = np.random.normal(size=N)
m_real = np.random.normal(loc=0.5, scale=0.2, size= N)
b_real = np.random.normal(loc=1.0, scale=0.2, size= N)
y = m_real*x + b_real

# variables and placeholders
# variables are used for weights and they change during training
m = tf.Variable(tf.random_normal([]))
b = tf.Variable(tf.random_normal([]))

x_holder = tf.placeholder(tf.float32, shape=[batch_size])
y_holder = tf.placeholder(tf.float32, shape=[batch_size])

# model
model = m * x_holder + b
loss = tf.reduce_mean(tf.pow(model - y_holder, 2))

# optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# run it in a session

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