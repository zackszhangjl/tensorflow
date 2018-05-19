# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:36:18 2018

@author: zhangjialiang
"""

import tensorflow as tf

input1=tf.placeholder(tf.float32)
input2=tf.placeholder(tf.float32)



output=tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))