import numpy as np

class weights:
    def __init__(self,input_hidden,hidden_output,hidden_hidden = None):
        weights.input_hidden = input_hidden
        weights.hidden_output = hidden_output
        weights.hidden_hidden = hidden_hidden 

flatten = lambda l: [item for sublist in l for item in sublist]

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

ReLU = lambda x: list(map(lambda x: x if x>0 else 0,x))
ReLU_Gradient = lambda x: list(map(lambda x: 1 if x>0 else 0,x))

LeakyReLU = lambda x,a: list(map(lambda x: x if x>0 else a*x,x))
LeakyReLU_GRadient = lambda x,a: list(map(lambda x: 1 if x>0 else a*1,x))


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


def FeedForward(inputPatterns,weightMatrix, architecture = {'hiddenLayers': 2,'respectiveHiddenUnits':[4,4]},returnType = 1):
    nlayers = architecture['hiddenLayers']
    hiddenUnits = architecture['respectiveHiddenUnits']
    w = weightMatrix.input_hidden
    b = []
    V = []
    b.append(np.add(np.dot(inputPatterns,w['weights']), w['bias']))
    output = Tanh(b[0])
    V.append(output)

    if(nlayers > 1):
        for n in range(0,nlayers-1):
            w = weightMatrix.hidden_hidden[n]
            b.append(np.add(np.dot(output,w['weights']), w['bias']))
            output = Tanh(b[n+1])
            V.append(output)
    
    w = weightMatrix.hidden_output
    b.append(np.add(np.dot(output,w['weights']), w['bias']))
    V.append(Tanh(b[nlayers]))
    if(returnType == 1):
        return V, b
    else:
        output = V.pop()
        return output


def GetGradients(outputs,b,weightMatrix,xCurrent,yCurrent):
    deltaW = []
    deltaB = []
    nb = len(b)-1
    w1 = weightMatrix.input_hidden    
    w2 = weightMatrix.hidden_hidden
    w3 = weightMatrix.hidden_output

    tempDelta = (yCurrent - outputs[nb])*Tanh_Gradient(b[nb])
    deltaB.append(np.sum(tempDelta,axis=0)/np.size(tempDelta,axis=0))
    deltaW.append(np.dot(np.transpose(outputs[nb-1]),tempDelta))
    
   
    tempDeltab =np.dot(np.transpose(np.dot(tempDelta,np.transpose(w3['bias']))),Tanh_Gradient(b[nb-1]))
    deltaB.append(tempDeltab)    

    tempDelta = np.dot(tempDelta,np.transpose(w3['weights']))*Tanh_Gradient(b[nb-1])
    deltaW.append(np.dot(np.transpose(outputs[nb-2]),tempDelta))   
    k = 2
    for n in range(len(w2)-1,-1,-1):
        w = w2[n]
        tempDeltab = np.dot(np.transpose(np.dot(tempDelta,np.transpose(w['bias']))),Tanh_Gradient(b[nb-k]))
        deltaB.append(tempDeltab) 
        tempDelta = np.dot(tempDelta,np.transpose(w['weights']))*Tanh_Gradient(b[nb-k])
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



    