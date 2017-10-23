import numpy as np
import nn_Utilities as nn
import copy
import csv 
import sys

def k_means(x, k=5, threshold=0.00001,batchSize = 10000):
    empty = ()
    nPatterns = np.size(x,axis=0)   
    inputDimensions = np.size(x,axis=1)    
    weightMatrix = np.random.uniform(np.min(x),np.max(x),(k, inputDimensions))
    tolerance = 1
    n = 0
    variance = 0
    iterations = 0
    while(iterations < 20):
        batch = np.arange(n,n+batchSize,1)
        xCurrent = x[batch]
        distance = nn.EucledianDistance(xCurrent,weightMatrix)
        clusterIndices = np.argmin(distance,axis = 1)
        clusters = dict()
        for i in range(0,batchSize,1):
            if clusterIndices[i] in clusters:
                clusters[clusterIndices[i]].append(batch[i])
            else:
                clusters[clusterIndices[i]] = [batch[i]]
        weightMatrixCopy = copy.deepcopy(weightMatrix)
        for i in range(0,k):
            currentCluster = np.array(clusters.get(i))
            if np.shape(currentCluster) is not empty:
                weightMatrix[i] = np.sum(x[currentCluster],axis=0)/len(currentCluster)

        
        n = n + batchSize
        # print(n)
        variance = variance + np.sum(distance)
        if(n == nPatterns):
            variance = np.sqrt(variance)            
            print('variance = ',variance)            
            iterations+= 1
            print(iterations)
            variance = 0
            n = 0
            
    return weightMatrix


 
f = open('activity/1.csv','r')
reader = csv.reader(f, delimiter=",")
x = list(reader)
result = np.array(x).astype("float")
f.close()

print(len(result))
print(np.shape(result))
x = result
w = k_means(x,k = 10)

def Kohonen_Network(x,outputShape = {'x': 10, 'y': None}):
    return 0

def NeuralGas(x,outputShape = {'x': 10, 'y': None}):
    return 0

