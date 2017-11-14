import numpy as np
import matplotlib.pyplot as plt
from numba import jit

class weights:
    def __init__(self,input_hidden,hidden_output,hidden_hidden = None):
        self.input_hidden = input_hidden
        self.hidden_output = hidden_output
        self.hidden_hidden = hidden_hidden 


flatten = lambda l: [item for sublist in l for item in sublist]

def softmax(x):
    x_exp = np.exp(np.array(x,dtype=float),dtype=float)
    x_exp_sum = np.sum(x_exp,axis = 1)
    return x_exp/x_exp_sum[:,None]

# Activation functions
def Sigmoid(x):
    xNumpy = np.array(x)
    return 1/(1+np.exp(-xNumpy))
def Sigmoid_Gradient(x):
    return Sigmoid(x)*(1 - Sigmoid(x))

def Tanh(x,beta=0.5):
    xNumpy = np.array(x)
    return np.tanh(beta*xNumpy)
def Tanh_Gradient(x,beta=0.5):
    return beta*(1-(np.square(Tanh(x,beta))))

def SoftPlus(x):
    xNumpy = np.array(x)
    return np.log(1 + np.exp(xNumpy))
def SoftPlus_Gradient(x):
    return Sigmoid(x)

def relu(x):
    xNumpy = np.array(x)
    xNumpy[xNumpy<0] = 0
    return xNumpy
def relu_grad(x):
    xNumpy = np.array(x)
    xNumpy[xNumpy>0] = 1
    xNumpy[xNumpy<=0] = 0
    return xNumpy

def leaky_relu(x,a=0.01):
    xNumpy = np.array(x,dtype=np.float64)
    xNumpy[xNumpy<0] = xNumpy[xNumpy<0]*a
    return xNumpy
def leaky_relu_grad(x,a=0.01):
    xNumpy = np.array(x,dtype=np.float64)
    xNumpy[xNumpy>0] = 1
    xNumpy[xNumpy<0] = a
    return xNumpy


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

ReLU = lambda x: list(map(lambda x: x if x>0 else 0,x))
ReLU_Gradient = lambda x: list(map(lambda x: 1 if x>0 else 0,x))

LeakyReLU = lambda x,a: list(map(lambda x: x if x>0 else a*x,x))
LeakyReLU_GRadient = lambda x,a: list(map(lambda x: 1 if x>0 else a*1,x))

def DynamicPlot(data1,data2 = None):
    plt.ion()
    plt.plot(data1,label = 'Training data')
    if data2 != None:
        plt.plot(flatten(data2),label = 'Validation data')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.legend()
    plt.pause(0.05)
    plt.clf()

def EucledianDistance(x, weightMatrix):
    w = weightMatrix    
    term1 = np.reshape(np.sum(np.square(w),axis=1),(1,-1)) + np.reshape(np.sum(np.square(x),axis=1),(-1,1))
    # shapeW = np.shape(w)
    # wTranspose = np.reshape(flatten(zip(*w)),(shapeW[1],shapeW[0]))
    term2 = 2*np.dot(x,np.transpose(w))
    eucledianDistance = term1 - term2      
    return eucledianDistance


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



    