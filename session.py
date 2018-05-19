# -*- coding: utf-8 -*-
"""
Created on Tue May 15 11:00:14 2018

@author: zhangjialiang
"""

import tensorflow as tf

matrix1=tf.constant([[3,3]])
matrix2=tf.constant([[2],
                    [2]])
product=tf.matmul(matrix1,matrix2)    #matrix multiply)

#method 1

sess=tf.Session()
result=sess.run(product)
print(result)
sess.close()