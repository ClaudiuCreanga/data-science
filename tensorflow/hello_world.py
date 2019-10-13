import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_dim=1))

model.compile(loss="mean_squared_error", optimizer="sgd")
x = np.array([-1, 0, 1, 2, 3, 4])
y = np.array([-3, -1, 1, 3, 5, 7])

model.fit(x, y, epochs = 500)

to_predict = np.array([10,11,12,13, 100])
print(model.predict(to_predict))