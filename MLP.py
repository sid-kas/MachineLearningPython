import numpy as np
import nn

def Train_MLP(inputPatterns, targetOutputs, eta = 0.01, architecture = {'hiddenLayers': 4,'respectiveHiddenUnits':[6,5,7,4]}, batchSize = 100, outputClasses = 1):
    checkData = 10
    updates = 10**4
    nPatterns = np.size(inputPatterns,axis = 0)
    inputDimensions = np.size(inputPatterns,axis = 1)
    weightMatrix = nn.Initialize_weights(inputDimensions,outputClasses,architecture)
    for n in range(0,updates):
        j = np.random.random_integers(0,nPatterns-1,batchSize)
        xCurrent = inputPatterns[j]
        yCurrent = targetOutputs[j]

        (outputs, b) = nn.FeedForward(xCurrent, weightMatrix, architecture)

        (deltaW, deltaB) = nn.GetGradients(outputs, b, weightMatrix, xCurrent, yCurrent)
        
        weightMatrix = nn.UpdateWeights(weightMatrix, deltaW, deltaB,eta) # To do: update weights with adam optimizer

        if n%checkData == 0:
            testOutput = nn.FeedForward(inputPatterns,weightMatrix,architecture,returnType = 2)
            H = np.array((targetOutputs-testOutput)**2).sum()/(2*nPatterns)
            print(n,' Upadtes completed out of', updates,', Energy: ',H)




x = np.random.rand(1000,3)
y = np.random.rand(1000,4)
Train_MLP(x,y,eta = 0.001,outputClasses = 4)

