import numpy as  np
import matplotlib.pyplot as plt
import pickle
from UnsupervisedLearning import Stochastic_K_means
from MultilayerPerceptronNetwork import Train_MLP, Test_MLP
import nn_Utilities as nn


def GetGaussian(weightMatrix, randomDataPoints):
    w = weightMatrix
    x = randomDataPoints
    eucledianDistance = np.array(np.reshape(np.sum(np.square(w),axis=1),(1,-1)) + np.reshape(np.sum(np.square(x),axis=1),(-1,1)) - 2*np.dot(x,np.transpose(w)),dtype=float)
    exponential = np.exp(-eucledianDistance/2,dtype=float)
    print(np.shape(exponential))
    expSum = np.sum(exponential,axis = 1)
    gaussianNeuronMatrix = exponential/expSum[:,None]
    print(np.shape(gaussianNeuronMatrix))
    return gaussianNeuronMatrix

# Data Intialization
# Read data
readFile = open("trainingDataInput.pickle","rb")
inputData = pickle.load(readFile)
readFile.close() 

readFile = open("trainingDataOutput.pickle","rb")
outputData = pickle.load(readFile)
readFile.close() 
flattenedout= nn.flatten(outputData)
targetOutput = np.reshape(flattenedout,(100000,200))
print(np.shape(targetOutput))

nPatterns = np.size(inputData)
nGaussianNeurons = 1200
inputDimensions = 64*64


w1 = Stochastic_K_means(inputData, dimensions = inputDimensions, k=nGaussianNeurons, checkData=10, batchSize = 1000)

outputUnsupervised = np.empty((nPatterns,nGaussianNeurons))
for n in range(0,nPatterns,1000):
    print(n)
    j = np.arange(n, n+1000,1)
    flattenedInput = nn.flatten(inputData[j])
    x = np.reshape(flattenedInput,(1000,inputDimensions))/255   
    outputUnsupervised[j,:] = GetGaussian(w,x)


w2 = Train_MLP(outputUnsupervised, targetOutput, architecture = {'hiddenLayers': 3,'respectiveHiddenUnits':[200,200,400]}, batchSize = 100,outputClasses = 200)







