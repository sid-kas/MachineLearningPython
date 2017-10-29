import numpy as np
import nn_Utilities as nn
import copy
import csv 
import itertools as it
import timeit
import cProfile

def Stochastic_K_means(inputData,inputDimensions,nClusters=5,updates=10**5,batchSize=100,checkData=5):
    x = inputData
    k = nClusters
    nPatterns = np.size(x,axis=0)   
    weightMatrix = np.random.uniform(np.min(x),np.max(x),(k, inputDimensions))
    empty = ()
    n = 0
    tolerance = 10**-5
    currentVariance = 1000
    variance = []
    clusters = dict()
    while(n< updates):
        batch = np.random.randint(0,nPatterns,batchSize)
        flattenedInput = nn.flatten(x[batch])
        xCurrent = np.reshape(flattenedInput,(batchSize,inputDimensions))
        distance = nn.EucledianDistance(xCurrent,weightMatrix)

        clusterIndices = np.argmin(distance,axis = 1)       
        for i in range(0,batchSize,1):
            if clusterIndices[i] in clusters:
                for j in range(0,k):
                    if j != clusterIndices[i] and j in clusters:
                        if batch[i] in clusters[j]:
                            clusters[j].remove(batch[i])
                    elif batch[i] not in clusters[clusterIndices[i]]:
                        clusters[clusterIndices[i]].append(batch[i])                
            else:
                clusters[clusterIndices[i]] = [batch[i]]
        weightMatrixCopy = copy.deepcopy(weightMatrix)

        patternsSum = 0
        for i in range(0,k):
            currentCluster = np.array(clusters.get(i))
            shape = np.shape(currentCluster)
            if shape is not empty:
                if shape[0] is not 0:
                    patternsSum = patternsSum + len(currentCluster)
                    flattenedInput = nn.flatten(x[currentCluster])
                    xCurrentCluster = np.reshape(flattenedInput,(len(currentCluster),inputDimensions))
                    weightMatrix[i] = np.sum(xCurrentCluster,axis=0)/len(currentCluster)
        n += 1
       
        if(n>5):
            currentVariance = (np.sum(np.abs(weightMatrix - weightMatrixCopy)))
            if(currentVariance< tolerance):
                print('Final Variance = ',currentVariance)
                break
            variance.append(currentVariance)
        if n%checkData == 0:
            print('Iterations:',n)
            print('Cheked patterns = ',patternsSum)
            print('Variance = ',currentVariance)
            # nn.DynamicPlot(variance)
    
    return weightMatrix

 
# f = open('activity/1.csv','r')
# reader = csv.reader(f, delimiter=",")
# x = list(reader)
# result = np.array(x).astype("float")
# f.close()

# print(len(result))
# print(np.shape(result))

# x = np.random.rand(50000,5000)
# cProfile.run('Stochastic_K_means(x,k = 1000, checkData = 50)')
# w = Stochastic_K_means(x,k = 100, checkData = 50)

def Kohonen_Network(x,outputShape = {'x': 10, 'y': None}):

    return 0

def NeuralGas(x,outputShape = {'x': 10, 'y': None}):
    return 0

