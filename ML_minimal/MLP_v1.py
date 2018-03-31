import numpy as np
import matplotlib.pyplot as plt
import pickle

class g:
      relu = 'relu'  
      tanh = 'tanh'
      sigmoid = 'sigmoid'
      leakyRelu = 'leaky_relu'

class weights:
    def __init__(self,input_hidden,hidden_output,hidden_hidden = None):
        self.input_hidden = input_hidden
        self.hidden_output = hidden_output
        self.hidden_hidden = hidden_hidden 

class activations:
    def __init__(self,type):
        if(type == 'sigmoid'):
            self.activation = Sigmoid
            self.activation_Grad = Sigmoid_Gradient
        if(type == 'tanh'):
            self.activation = Tanh
            self.activation_Grad = Tanh_Gradient
        if(type == 'relu'):
            self.activation = relu
            self.activation_Grad = relu_grad
        if(type == 'leaky_relu'):
            self.activation = leaky_relu
            self.activation_Grad = leaky_relu_grad

def Train_MLP(trainingData, targetOutputs, activation = g.tanh , updates = 2*10**3, validationData = None, eta = 0.01, architecture = {'hiddenLayers': 4,'respectiveHiddenUnits':[6,5,7,4]}, batchSize = 100, outputClasses = 1):
    checkData = 100
    nPatterns = np.size(trainingData,axis = 0)
    inputDimensions = np.size(trainingData,axis = 1)
    weightMatrix = Initialize_weights(inputDimensions,outputClasses,architecture)
    energyTraining = []
    energyValiation = []
    for n in range(0,updates):
        j = np.random.random_integers(0,nPatterns-1,batchSize)
        xCurrent = trainingData[j]
        yCurrent = targetOutputs[j]

        (outputs, b) = FeedForward(xCurrent, weightMatrix, architecture, activationFunction = activation)

        (deltaW, deltaB) = GetGradients(outputs, b, weightMatrix, xCurrent, yCurrent, activationFunction = activation)
        
        weightMatrix = UpdateWeights(weightMatrix, deltaW, deltaB,eta) # To do: update weights with adam optimizer

        if n%checkData == 0:
            testOutput = FeedForward(trainingData,weightMatrix,architecture,returnType = 2)
            H = np.array((targetOutputs-testOutput)**2).sum()/(2*nPatterns)
            energyTraining.append(H)
            if validationData is not None:
                validOutput = FeedForward(validationData,weightMatrix,architecture,returnType = 2)
                energyValiation.append(np.array((targetOutputs-validOutput)**2).sum()/(2*nPatterns))
            print(n,' Upadtes completed out of', updates,', Energy: ',H)
    return weightMatrix


def Test_MLP(testPatterns,testOutputs,weightMatrix,architecture = {'hiddenLayers': 4,'respectiveHiddenUnits':[6,5,7,4]}):
    nPatterns = np.size(testPatterns,axis = 0)
    output_FeedForward = FeedForward(testPatterns,weightMatrix,architecture,returnType = 2)
    H = np.array((testOutputs-output_FeedForward)**2).sum()/(2*nPatterns)
    print('Energy: ',H)
    return H


def Initialize_weights(inputDimensions,outputClasses,architecture = {'hiddenLayers': 2,'respectiveHiddenUnits':[4,4]}):
    nlayers = architecture['hiddenLayers']
    hiddenUnits = architecture['respectiveHiddenUnits']
    input_hidden = {'weights' : np.random.uniform(-1,1,(inputDimensions, hiddenUnits[0])), 
                    'bias': np.random.uniform(-1,1,(1, hiddenUnits[0]))}
    hidden_output = {'weights' : np.random.uniform(-1,1,(hiddenUnits[nlayers-1], outputClasses)),
                      'bias' : np.random.uniform(-1,1,(1, outputClasses))}
    
    if(nlayers > 1):
        hidden_hidden = np.empty((nlayers-1),dtype=object)
        for n in range(0,nlayers-1):
            hidden_hidden[n] = {'weights' : np.random.uniform(-1,1,(hiddenUnits[n],hiddenUnits[n+1])),
                                 'bias' :  np.random.uniform(-1,1,(1,hiddenUnits[n+1]))}
        weightMatrix = weights(input_hidden,hidden_output,hidden_hidden)
        return weightMatrix
    else:
        weightMatrix = weights(input_hidden,hidden_output)
        return weightMatrix


def FeedForward(inputPatterns,weightMatrix, architecture = {'hiddenLayers': 2,'respectiveHiddenUnits':[4,4]},activationFunction = 'tanh',returnType = 1):
    g = activations(activationFunction)
    nlayers = architecture['hiddenLayers']
    hiddenUnits = architecture['respectiveHiddenUnits']
    w = weightMatrix.input_hidden
    b = []
    V = []
    b.append(np.add(np.dot(inputPatterns,w['weights']), w['bias']))
    output = g.activation(b[0])
    V.append(output)

    if(nlayers > 1):
        for n in range(0,nlayers-1):
            w = weightMatrix.hidden_hidden[n]
            b.append(np.add(np.dot(output,w['weights']), w['bias']))
            output = g.activation(b[n+1])
            V.append(output)
    
    w = weightMatrix.hidden_output
    b.append(np.add(np.dot(output,w['weights']), w['bias']))
    if(activationFunction == 'relu' or activationFunction == 'leaky_relu'):
        V.append(g.activation(b[nlayers]))
    else:
        V.append(g.activation(b[nlayers]))
    
    if(returnType == 1):
        return V, b
    else:
        output = V.pop()
        return output


def GetGradients(outputs,b,weightMatrix,xCurrent,yCurrent, activationFunction = 'tanh'):
    g = activations(activationFunction)
    deltaW = []
    deltaB = []
    nb = len(b)-1
    w1 = weightMatrix.input_hidden    
    w2 = weightMatrix.hidden_hidden
    w3 = weightMatrix.hidden_output

    tempDelta = (yCurrent - outputs[nb])*g.activation_Grad(b[nb])
    deltaB.append(np.sum(tempDelta,axis=0)/np.size(tempDelta,axis=0))
    deltaW.append(np.dot(np.transpose(outputs[nb-1]),tempDelta))
    
   
    tempDeltab =np.dot(np.transpose(np.dot(tempDelta,np.transpose(w3['bias']))),g.activation_Grad(b[nb-1]))
    deltaB.append(tempDeltab)    

    tempDelta = np.dot(tempDelta,np.transpose(w3['weights']))*g.activation_Grad(b[nb-1])
    deltaW.append(np.dot(np.transpose(outputs[nb-2]),tempDelta))   
    k = 2
    for n in range(len(w2)-1,-1,-1):
        w = w2[n]
        tempDeltab = np.dot(np.transpose(np.dot(tempDelta,np.transpose(w['bias']))),g.activation_Grad(b[nb-k]))
        deltaB.append(tempDeltab) 
        tempDelta = np.dot(tempDelta,np.transpose(w['weights']))*g.activation_Grad(b[nb-k])
        if(nb-k-1 >=0):
            deltaW.append(np.dot(np.transpose(outputs[nb-k-1]),tempDelta))     
        else:
            deltaW.append(np.dot(np.transpose(xCurrent),tempDelta))  
        k += 1

    return deltaW, deltaB


def UpdateWeights(weightMatrix, deltaW, deltaB,eta = 0.01):
    w = []
    w.append(weightMatrix.input_hidden)    
    for item in weightMatrix.hidden_hidden:
        w.append(item)
    w.append(weightMatrix.hidden_output)
    w.reverse()

    for n in range(0,len(deltaW)):
        wCurrent = w[n]
        wCurrent['weights'] = wCurrent['weights'] + eta*deltaW[n]
        wCurrent['bias'] = wCurrent['bias'] + eta*deltaB[n]
    
    input_hidden = w.pop()
    hidden_hidden = []
    for n in range(0,len(w)-1,1):
        hidden_hidden.append(w.pop())
    hidden_output = w.pop()

    weightMatrix = weights(input_hidden,hidden_output,hidden_hidden)
    return weightMatrix