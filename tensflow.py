#import tensorflow as tf
import numpy as np

w = np.ones((10,3))

j = [1,3,1]
k = w[j]
deltaW = 0.1 * (1.5-w[j])
for i in range(0,3):
    w[j[i]] = w[j[i]] + deltaW[i] 



print(k)





# def foo(g):
#     l = tf.reduce_sum(tf.square(g),axis = 0)
#     return l

# z = tf.Variable(tf.zeros(1,5))

# with tf.Session() as test:
#     x = tf.constant(np.random.rand(5,10))
#     y = tf.constant(np.random.rand(10,5))
#     test.run(tf.global_variables_initializer())
#     for i in range(1,10):
#         k = tf.matmul(x,y)*i
#         g = k + 2
#         z = foo(g)
#         tf.assign(z[2],100)
#         output = test.run(z)
#         print(output)
