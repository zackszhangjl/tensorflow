# -*- coding: utf-8 -*-
"""
Created on Tue May 15 20:45:41 2018

@author: zhangjialiang
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:48:59 2018

@author: zhangjialiang
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:09:49 2018

@author: zhangjialiang
"""

import tensorflow as tf


def add_layer(inputs,in_size,out_size,activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            Weights=tf.Variable(tf.random_normal([in_size,out_size]),name='W')
        with tf.name_scope('biases'):
            biases=tf.Variable(tf.zeros([1,out_size])+0.1,name='b')
        with tf.name_scope('Wx_plus_b'):
                Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs


with tf.name_scope('inputs'):
    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
    ys=tf.placeholder(tf.float32,[None,1],name='y_input')


l1=add_layer(xs,1,10,activation_function=tf.nn.relu)
prediction=add_layer(l1,10,1,activation_function=None)


with tf.name_scope('loss'):
    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))
with tf.name_scope('train'):
    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#init=tf.initialize_all_variables()
sess=tf.Session()
writer=tf.summary.FileWriter("logs/",sess.graph)
#sess.run(init)
sess.run(tf.initialize_all_variables())


