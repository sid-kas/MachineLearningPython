import numpy as np
import matplotlib.pyplot as plt
import nn_Utilities as nn

def Train_MLP(trainingData, targetOutputs, validationData = None, eta = 0.01, architecture = {'hiddenLayers': 4,'respectiveHiddenUnits':[6,5,7,4]}, batchSize = 100, outputClasses = 1):
    checkData = 10
    updates = 10**5
    nPatterns = np.size(trainingData,axis = 0)
    inputDimensions = np.size(trainingData,axis = 1)
    weightMatrix = nn.Initialize_weights(inputDimensions,outputClasses,architecture)
    energyTraining = []
    energyValiation = []
    for n in range(0,updates):
        j = np.random.random_integers(0,nPatterns-1,batchSize)
        xCurrent = trainingData[j]
        yCurrent = targetOutputs[j]

        (outputs, b) = nn.FeedForward(xCurrent, weightMatrix, architecture)

        (deltaW, deltaB) = nn.GetGradients(outputs, b, weightMatrix, xCurrent, yCurrent)
        
        weightMatrix = nn.UpdateWeights(weightMatrix, deltaW, deltaB,eta) # To do: update weights with adam optimizer

        if n%checkData == 0:
            testOutput = nn.FeedForward(trainingData,weightMatrix,architecture,returnType = 2)
            H = np.array((targetOutputs-testOutput)**2).sum()/(2*nPatterns)
            energyTraining.append(H)
            if validationData is not None:
                validOutput = nn.FeedForward(validationData,weightMatrix,architecture,returnType = 2)
                energyValiation.append(np.array((targetOutputs-validOutput)**2).sum()/(2*nPatterns))
                nn.DynamicPlot(energyTraining)
            print(n,' Upadtes completed out of', updates,', Energy: ',H)
    return weightMatrix


def Test_MLP(testPatterns,testOutputs,weightMatrix,architecture = {'hiddenLayers': 4,'respectiveHiddenUnits':[6,5,7,4]}):
    nPatterns = np.size(testPatterns,axis = 0)
    output_FeedForward = nn.FeedForward(testPatterns,weightMatrix,architecture,returnType = 2)
    H = np.array((testOutputs-output_FeedForward)**2).sum()/(2*nPatterns)
    print('Energy: ',H)
    return H


