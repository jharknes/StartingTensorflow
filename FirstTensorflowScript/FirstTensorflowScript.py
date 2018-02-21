# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf

deep_learning = tf.constant('Deep Learing')
session = tf.Session()
print(session.run(deep_learning))

a = tf.constant(2)
b = tf.constant(3)
multiply = tf.multiply(a,b)
print(session.run(multiply))

#variable introduction
weights = tf.Variable(tf.random_normal([300,200],stddev=.5), name="weights")
"""two arguments passed to tf.Variable. First argument produces a tensor initialized
 with a normal distribution and a standard deviation of .5. The tensor is of 
size 300 x 200, meaning the weights connect a layer of 300 neurons to a layer
 of 200 neurons. The name portion is what it sounds like."""

"""weights is meant to be trainable, meaning we will automatically compute and 
apply gradients to it. If weights should not be trainable, an oprion flag can 
be added on. ie. 
weights = tf.Variable(tf.random_normal([300,200],stdev=.5), name="weights", trainable=False)"""

#placeholders
x = tf.placeholder(tf.float32, name="x", shape=[None,784])
w = tf.Variable(tf.random_uniform([784,10], -1,1),name="w")
multiply = tf.matmul(x,w)

"""x represents a minibatch of data stored as floatswith 784 columns, meaning
 each data sample has 784 dimensions. The number of rows is undefined which means 
x can initialized with an arbitrary number of data samples. Doing the above 
code allows us to multiply everything in the tensor at once by w instead of one
at a time because x is a full minibatch."""
