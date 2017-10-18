import numpy as np
import nn_Utilities as nn
import copy

def k_means(x, k=5, threshold=0.00001):
    nPatterns = np.size(x,axis=0)   
    inputDimensions = np.size(x,axis=1)    
    weightMatrix = np.random.uniform(np.min(x),np.max(x),(k, inputDimensions))
    tolerance = 1
    while(tolerance > threshold):
        distance = nn.EucledianDistance(x,weightMatrix)
        clusterIndices = np.argmin(distance,axis = 1)
        clusters = dict()
        for n in range(0,nPatterns):
            if clusterIndices[n] in clusters:
                clusters[clusterIndices[n]].append(x[n])
            else:
                clusters[clusterIndices[n]] = [x[n]]
        weightMatrixCopy = copy.deepcopy(weightMatrix)
        for n in range(0,k):
            currentCluster = clusters.get(n)
            reshapedCluster = np.reshape(nn.flatten(currentCluster),(len(currentCluster),inputDimensions))
            weightMatrix[n] = np.sum(reshapedCluster,axis=0)/len(currentCluster)

        tolerance = np.abs(np.sum((weightMatrix - weightMatrixCopy)))/(k*inputDimensions)
        print('tolerance = ',tolerance)
    return weightMatrix
   
def Kohonen_Network(x,outputShape = {'x': 10, 'y': None}):
    return 0

def NeuralGas(x,outputShape = {'x': 10, 'y': None}):
    return 0

