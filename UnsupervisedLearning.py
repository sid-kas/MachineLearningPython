import numpy as np
import nn_Utilities as nn
import copy
import csv 
import itertools as it

def Stochastic_K_means(x, k=5, checkData = 5, batchSize = 1000):
    empty = ()
    nPatterns = np.size(x,axis=0)   
    inputDimensions = np.size(x,axis=1)    
    weightMatrix = np.random.uniform(np.min(x),np.max(x),(k, inputDimensions))
    tolerance = 1
    n = 0
    variance = []
    iterations = 0
    clusters = dict()
    while(iterations < 20):
        batch = np.random.randint(0,nPatterns,batchSize)
        xCurrent = x[batch]
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
            if np.shape(currentCluster) is not empty:
                patternsSum = patternsSum + len(currentCluster)
                weightMatrix[i] = np.sum(x[currentCluster],axis=0)/len(currentCluster)
        n += 1
        print('Iterations:',n)
        print('Cheked patterns = ',patternsSum)
        if(n>3):
            variance.append(int(np.sum(np.abs(weightMatrix - weightMatrixCopy))))
        if n%checkData == 0:
            nn.DynamicPlot(variance)
    
    return weightMatrix

it.
 
f = open('activity/1.csv','r')
reader = csv.reader(f, delimiter=",")
x = list(reader)
result = np.array(x).astype("float")
f.close()

print(len(result))
print(np.shape(result))
x = result
w = Stochastic_K_means(x,k = 1000)

def Kohonen_Network(x,outputShape = {'x': 10, 'y': None}):

    return 0

def NeuralGas(x,outputShape = {'x': 10, 'y': None}):
    return 0

