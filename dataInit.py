import numpy as np
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as img
from os import listdir
from os.path import isfile, join






raw = np.loadtxt('tiny-imagenet-200/wnids.txt',dtype=str)
file_list = []
for n in raw:
    text = n[2:]
    text = text[:9]
    file_list.append(text)

matrix = np.identity(200)



images1 = np.empty((500*200,1), dtype=object)
images2 = np.empty(500*200, dtype=object)
images3 = np.empty(500*200, dtype=object)
nold = 0
count = 0
for file in raw:
    mypath='tiny-imagenet-200/train/'+ file + '/images/'
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    print(file)
    out = matrix[count]
    for n in range(0, len(onlyfiles)):
        k = nold + n
        imgMatrix = cv2.imread( join(mypath,onlyfiles[n]), cv2.IMREAD_GRAYSCALE )
        inputVec = np.array(np.reshape(imgMatrix,(64*64,1))).flatten()
        dict = {'Name': str(file),'Input':inputVec,'Output':out}
        
        images1[k] = dict
        images2[k] = inputVec
        images3[k] = out

    nold = nold + n + 1
    count += 1

#save data
saveArray = open("trainingDataDict.pickle","wb")
pickle.dump(images1, saveArray)
saveArray.close()

saveArray = open("trainingDataInput.pickle","wb")
pickle.dump(images2, saveArray)
saveArray.close()
saveArray = open("trainingDataOutput.pickle","wb")
pickle.dump(images3, saveArray)
saveArray.close()

#read data
# f2 = open("trainingDataDict.pickle","rb")
# dataread = pickle.load(f2)
# f2.close() 




#img = cv2.imread('bread'+'/'+str(1) + '.jpg', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('bread'+'/'+str(1) + '.jpg', cv2.IMREAD_COLOR)
# cv2.imshow('frame',img2)
# cv2.waitKey()
# cv2.destroyAllWindows()

# kf = np.array(img)
# l = np.shape(kf)
# vec = np.reshape(kf, (l[0]*l[1],1))


# k = 1



# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# n_nodes_hl1 = 500
# n_nodes_hl2 = 500
# n_nodes_hl3 = 500

# n_classes = 10
# batch_size = 100

# x = tf.placeholder('float', [None, 784])
# y = tf.placeholder('float')

# def neural_network_model(data):
#     hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
#                       'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

#     hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
#                       'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

#     hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
#                       'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

#     output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
#                     'biases':tf.Variable(tf.random_normal([n_classes])),}


#     l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
#     l1 = tf.nn.relu(l1)

#     l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
#     l2 = tf.nn.relu(l2)

#     l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
#     l3 = tf.nn.relu(l3)

#     output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

#     return output

# def train_neural_network(x):
#     prediction = neural_network_model(x)
#     # OLD VERSION:
#     #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
#     # NEW:
#     cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
#     optimizer = tf.train.AdamOptimizer().minimize(cost)
    
#     hm_epochs = 10
#     with tf.Session() as sess:
#         # OLD:
#         #sess.run(tf.initialize_all_variables())
#         # NEW:
#         sess.run(tf.global_variables_initializer())

#         for epoch in range(hm_epochs):
#             epoch_loss = 0
#             for _ in range(int(mnist.train.num_examples/batch_size)):
#                 epoch_x, epoch_y = mnist.train.next_batch(batch_size)
#                 _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
#                 epoch_loss += c

#             print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

#         correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

#         accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#         print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

# train_neural_network(x)