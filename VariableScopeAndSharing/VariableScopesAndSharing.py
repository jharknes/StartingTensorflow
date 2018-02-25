#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 09:35:19 2018

@author: josh
"""

import tensorflow as tf

def my_network(input):
    w1 = tf.Variable(tf.random_uniform([784, 100], -1, 1), name = 'w1')
    b1 = tf.Variable(tf.zeros([100]), name = 'biases1')
    output1 = tf.matmul(input, w1) + b1
    
    w2 = tf.Variable(tf.random_uniform([100, 50], -1, 1), name = 'w2')
    b2 = tf.Variable(tf.zeros([50]), name = 'biases2')
    output2 = tf.matmul(output1, w2) + b2
    
    w3 = tf.Variable(tf.random_uniform([50, 10], -1, 1), name = 'w3')
    b3 = tf.Variable(tf.zeros([10]), name = 'biases3')
    output3 = tf.matmul(output2, w3) + b3
    
    print('Printing names of weight parameters.')
    print(w1.name, w2.name, w3.name)
    print('Printing names of bias parameters')
    print(b1.name, b2.name, b3.name)
    
    return output3

"""The above code leads tensorflow into making new variables each time the function 
is run. To avoid this, we should not use the Variable function in tensorflow,
but we should either get_variable or varable_scope. The correct code is written below.
The only issue with the below code is that when two different inputs are used in the 
same kernel session, the variables after the first session will not be saved due to over sharing"""

def layer(input, weightShape, biasShape):
    weightInit = tf.random_uniform_initializer(minval = -1, maxval = 1)
    biasInit = tf.constant_initializer(value=0)
    w = tf.get_variable('w', weightShape, initializer = weightInit)
    b = tf.get_variable('b', biasShape, initializer = biasInit)
    return tf.matmul(input, w) + b
    
def my_network2(input):
    with tf.variable_scope('layer_1'):
        output1b = layer(input, [784, 100], [100])
        
    with tf.variable_scope('layer_2'):
        output2b = layer(output1b, [100, 50], [50])
        
    with tf.variable_scope('layer_3'):
        output3b = layer(output2b, [50, 10], [10])
        
    return output3b


'''The below code allows the same variable to be used within the same scope. This 
allows variables to be overwritten and shared.'''

with tf.variable_scope("shared_variables") as scope:
    scope.reuse_variables()
    i1 = tf.placeholder(tf.float32, [1000, 784], name='i1')
    my_network2(i1)
    scope.reuse_variables()
    i2 = tf.placeholder(tf.float32, [1000, 784], name='i2')
    my_network(i2)