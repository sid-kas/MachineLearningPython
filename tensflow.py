import tensorflow as tf
import numpy as np

def foo(g):
    l = tf.reduce_sum(tf.square(g),axis = 0)
    return l

z = tf.Variable(tf.zeros(1,5))

with tf.Session() as test:
    x = tf.constant(np.random.rand(5,10))
    y = tf.constant(np.random.rand(10,5))
    test.run(tf.global_variables_initializer())
    for i in range(1,10):
        k = tf.matmul(x,y)*i
        g = k + 2
        z = foo(g)
        tf.assign(z[2],100)
        output = test.run(z)
        print(output)
