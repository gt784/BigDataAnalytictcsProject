from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random

# Parameters

training_epochs = 180
display_step = 60

# Training Data
tr_X = numpy.asarray([0.538,0.469,0.469,0.458,0.458,0.458,0.524,0.524,0.524,0.524,0.524,0.524,0.524,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.538,0.499,0.499,0.499,0.499,0.428,0.428,0.448,0.448,0.448,0.448,0.448,0.448,0.448,0.448,0.448] )
tr_Y = numpy.asarray([6.575,6.421,7.185,6.998,7.147,6.43,6.012,6.172,5.631,6.004,.377,6.009,5.889,5.949,6.096,5.834,5.935,5.99,5.456,5.727,5.57,5.965,6.142,5.813,5.924,5.599,5.813,6.047,6.495,6.674,5.713,6.072,5.95,5.701,6.096,5.933,5.841,5.85,5.966,6.595,7.024,6.77,6.169,6.211,6.069,5.682,5.786,6.03,5.399,5.602])
n_samples = tr_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
predict = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(predict-Y, 2))/(2*n_samples)
# Gradient descent
train_optimizer = tf.train.GradientDescentOptimizer(0.03).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(tr_X, tr_Y):
            sess.run(train_optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: tr_X, Y:tr_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Completed!")
    training_cost = sess.run(cost, feed_dict={X: tr_X, Y: tr_Y})
    print("Train cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(tr_X, tr_Y, 'ro', label='Original data')
    plt.plot(tr_X, sess.run(W) * tr_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([0.448,0.439,0.439,0.439,0.439,0.41,0.403,0.41,0.411,0.453,0.453,0.453,0.453,0.453,0.453,0.4161,0.398,0.398,0.409,0.409,0.409,0.413,0.413,0.413,0.413,0.437,0.437,0.437,0.437,0.437,0.437,0.426,0.426,0.426,0.426,0.449,0.449,0.449,0.449,0.489,0.489,0.489,0.489,0.464,0.464,0.464,0.445,0.445,0.445,0.445,])
    test_Y = numpy.asarray([5.963,6.115,6.511,5.998,5.888,7.249,6.383,6.816,6.145,5.927,5.741,5.966,6.456,6.762,7.104,6.29,5.787,5.878,5.594,5.885,6.417,5.961,6.065,6.245,6.273,6.286,6.279,6.14,6.232,5.874,6.727,6.619,6.302,6.167,6.389,6.63,6.015,6.121,7.007,7.079,6.417,6.405,6.442,6.211,6.249,6.625,6.163,8.069,7.82,7.416])
    print("Testing... (Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(predict - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Test cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(tr_X, sess.run(W) * tr_X + sess.run(b), label='Fitted line')
    plt.legend()
plt.show()