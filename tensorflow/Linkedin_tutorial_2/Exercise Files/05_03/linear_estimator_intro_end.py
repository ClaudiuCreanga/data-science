''' Demonstrate how estimators can be used for regression '''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# Define constants
N = 1000
num_steps = 800

# Step 1: Generate input points
x_train = np.random.normal(size=N)
m = np.random.normal(loc=0.5, scale=0.2, size=N)
b = np.random.normal(loc=1.0, scale=0.2, size=N)
y_train = m * x_train + b

# Step 2: Create a feature column
x_col = tf.feature_column.numeric_column('x_coords')

# Step 3: Create a LinearRegressor

# Step 4: Train the estimator with the generated data

# Step 5: Predict the y-values when x equals 1.0 and 2.0