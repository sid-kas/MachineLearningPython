import numpy as np
import nn

def Train_MLP(inputPatterns, targetOutputs, eta = 0.01, architecture = {'hiddenLayers': 4,'respectiveHiddenUnits':[6,5,7,4]}, batchSize = 100, outputClasses = 1):
    checkData = 10**2
    updates = 10**5
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
    return weightMatrix


def Test_MLP(testPatterns,testOutputs,weightMatrix,architecture = {'hiddenLayers': 4,'respectiveHiddenUnits':[6,5,7,4]}):
    nPatterns = np.size(testPatterns,axis = 0)
    output_FeedForward = nn.FeedForward(testPatterns,weightMatrix,architecture,returnType = 2)
    H = np.array((testOutputs-output_FeedForward)**2).sum()/(2*nPatterns)
    print('Energy: ',H)
    return H


