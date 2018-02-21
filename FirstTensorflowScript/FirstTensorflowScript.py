# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
deep_learning = tf.constant('Deep Learing')
session = tf.Session() #a class used to run tensorflow operations
print(session.run(deep_learning))
a = tf.constant(2)
b = tf.constant(3)
multiply = tf.multiply(a,b)
print(session.run(multiply))