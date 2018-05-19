# -*- coding: utf-8 -*-
"""
Created on Tue May 15 11:00:14 2018

@author: zhangjialiang
"""

import tensorflow as tf

state=tf.Variable(0,name='counter')
#print(state.name)
one=tf.constant(1)
print(state)
new_value=tf.add(state,one)
update=tf.assign(state,new_value)

init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))