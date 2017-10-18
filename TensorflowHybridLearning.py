import numpy as  np
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf


# Function definitions

def GetGaussian(weightMatrix, randomDataPoints):
    w = weightMatrix
    x = randomDataPoints
    eucledianDistance = np.reshape(np.sum(np.square(w),axis=1),(1,-1)) + np.reshape(np.sum(np.square(x),axis=1),(-1,1)) - 2*np.dot(x,np.transpose(w))
    exponential = np.exp(-eucledianDistance/2)
    expSum = np.sum(exponential,axis = 1)
    gaussianNeuronMatrix = exponential/expSum[:,None]
    return gaussianNeuronMatrix

# Read data
readFile = open("trainingDataInput.pickle","rb")
inputData = pickle.load(readFile)
readFile.close() 

readFile = open("trainingDataOutput.pickle","rb")
outputData = pickle.load(readFile)
readFile.close() 

# readFile = open("trainingDataDict.pickle","rb")
# trainingData = pickle.load(readFile)
# readFile.close() 

# Control parameters
nGaussianNeurons = 800
nPatterns = np.size(inputData)
inputDimensions = 64*64
etaU = 0.02
nUpdatesU = 5*10**2
checkData = 10
batchSize = 1000

# Unsupervised simple competetive learning

weightMatrixU = np.random.uniform(0,1,(nGaussianNeurons,inputDimensions))

flatten = lambda l: [item for sublist in l for item in sublist]

for nUnsupervised in range(0,nUpdatesU):
    j = np.random.random_integers(0,nPatterns-1,batchSize)
    flattenedInput = flatten(inputData[j])
    x = np.reshape(flattenedInput,(batchSize,inputDimensions))/255

    gaussianNeuronMatrix = GetGaussian(weightMatrixU,x)
    winningNeuronIndex = np.argmax(gaussianNeuronMatrix,axis=1)

    deltaW = etaU * (x-weightMatrixU[winningNeuronIndex])
    weightMatrixU[winningNeuronIndex] = weightMatrixU[winningNeuronIndex] + deltaW 

    if(nUnsupervised%checkData == 0):
        print("n = ", nUnsupervised)


outputUnsupervised = np.empty((nPatterns,nGaussianNeurons))
for n in range(0,nPatterns,1000):
    print(n)
    j = np.arange(n, n+1000,1)
    flattenedInput = flatten(inputData[j])
    x = np.reshape(flattenedInput,(batchSize,inputDimensions))/255   
    outputUnsupervised[j,:] = GetGaussian(weightMatrixU,x)


# Control parameters

etaS = 0.1
beta = 0.5
epsilon = 0
nUpdatesS = 3*10**5
checkData = 100
batchSize = 100
outputClasses = 200

# Supervised multilayer percertron learning
inputSupervised = outputUnsupervised
targetOutput = np.reshape(flatten(outputData),(nPatterns,outputClasses))


n_nodes_hl1 = 1200
n_nodes_hl2 = 1200
n_nodes_hl3 = 1200

n_classes = outputClasses
batch_size = batchSize

x = tf.placeholder('float', [None, nGaussianNeurons])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([nGaussianNeurons, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 25
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(nPatterns/batch_size)):
                j = np.random.random_integers(0,nPatterns-1,batchSize)
                epoch_x = inputSupervised[j]
                epoch_y = targetOutput[j]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:inputSupervised, y: targetOutput}))

train_neural_network(x)