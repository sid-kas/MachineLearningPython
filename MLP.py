import numpy as np
import nn


def Train_MLP(inputPatterns, targetOutputs, architecture = {'hiddenLayers': 4,'respectiveHiddenUnits':[4,5,7,4]},batchSize = 10, outputClasses = 1):
    nPatterns = np.size(inputPatterns,axis = 0)
    inputDimensions = np.size(inputPatterns,axis = 1)
    weightMatrix = nn.Initialize_weights(inputDimensions,outputClasses,architecture)
    for n in range(0,10):
        j = np.random.random_integers(0,nPatterns-1,batchSize)
        xCurrent = inputPatterns[j]
        yCurrent = targetOutputs[j]

        (outputs, b) = nn.FeedForward(xCurrent, weightMatrix, architecture)



x = np.random.rand(100,3)
y = np.random.rand(100,1)
Train_MLP(x,y)

