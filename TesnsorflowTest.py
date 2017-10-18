import numpy as  np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle


# Data Intialization
# Read data
readFile = open("trainingDataInput.pickle","rb")
inputData = pickle.load(readFile)
readFile.close() 

readFile = open("trainingDataOutput.pickle","rb")
outputData = pickle.load(readFile)
readFile.close() 

readFile = open("trainingDataDict.pickle","rb")
trainingData = pickle.load(readFile)
readFile.close() 



# Control parameters
nGaussianNeurons = 400
nPatterns = np.size(trainingData)
inputDimensions = 64*64
etaU = 0.02
etaS = 0.1
beta = 0.5
epsilon = 0
nUpdatesU = 10**8
nUpdatesS = 3*10**3
checkData = 10**2



# Unsupervised simple competetive learning

# Model parameters
#weightMatrixU = tf.Variable( tf.random_uniform((nGaussianNeurons,inputDimensions),0,1,dtype=tf.float64))
weightMatrixU = np.random.uniform(0,1,(nGaussianNeurons,inputDimensions))
neuronsIndices = tf.constant(np.reshape(np.arange(0,nGaussianNeurons,1),(-1,1)))

# Place holders

x_PH = tf.placeholder(dtype = tf.float64,shape = (1,inputDimensions))
weightMatrixU_PH = tf.placeholder(dtype = tf.float64,shape = (nGaussianNeurons,inputDimensions))
etaU_PH = tf.placeholder(dtype = tf.float64)

# Function definitions

def GetOutput(weightMatrixU_PH, x_PH, etaU_PH):

    gaussianNeuronMatrix = tf.nn.softmax(tf.sqrt(tf.reduce_sum(tf.square(weightMatrixU_PH-x_PH),axis=1)))
    winningNeuronIndex = tf.argmax(gaussianNeuronMatrix)
    deltaW = etaU_PH * (x_PH-weightMatrixU_PH[winningNeuronIndex])
    return winningNeuronIndex, deltaW



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for n in range(0, nUpdatesU):
        j = np.random.random_integers(0,nPatterns-1,1)
       
        x = np.array(np.reshape(inputData[j][0],(1,-1))/255)
        output = GetOutput(weightMatrixU_PH, x_PH, etaU_PH)
        (winningNeuronIndex, deltaW) = sess.run(output, feed_dict={weightMatrixU_PH : weightMatrixU, x_PH : x,etaU_PH : etaU})        
        
        weightMatrixU[winningNeuronIndex] = weightMatrixU[winningNeuronIndex] + deltaW
        
        if(n%checkData == 0):
            print(winningNeuronIndex)
            print("n = ", n)

