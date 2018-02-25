#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:27:13 2018

@author: josh
"""

import tensorflow as tf
from read_data import get_minibatch

x = tf.placeholder(tf.float32, name="x", shape=[None, 784])
w = tf.Variable(tf.random_uniform([784,10], -1, 1), name = "w")
b = tf.Variable(tf.zeroes([10]), name = "biases")
output = tf.matmul(x,w) + b

init_op = tf.initialize_all_variables()     #this is required to assign the variables

sess = tf.Session()
sess.run(init_op)                         #the initialization
feed_dict = {"x" : get_minibatch()}      #fills the placeholders with the necessary input data
sess.run(output, feed_dict=feed_dict)    

"""This code does not run. The code provided from the text does not 
provide a read_data library which describes the get_minibatch() function. This also
makes it a little more difficult to understand exactly what the last seesion actually
runs."""