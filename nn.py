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

def FeedForward(inputPatterns,weightMatrix, architecture = {'hiddenLayers': 2,'respectiveHiddenUnits':[4,4]}):
    nlayers = architecture['hiddenLayers']
    hiddenUnits = architecture['respectiveHiddenUnits']
    w = weightMatrix.input_hidden
    b = []
    b.append(np.add(np.dot(inputPatterns,w['weights']), w['bias']))
    V = Tanh(b[0])

    if(nlayers > 1):
        for n in range(0,nlayers-1):
            w = weightMatrix.hidden_hidden[n]
            b.append(np.add(np.dot(V,w['weights']), w['bias']))
            V = Tanh(b[n+1])
    
    w = weightMatrix.hidden_output
    b.append(np.add(np.dot(V,w['weights']), w['bias']))
    output = Tanh(b[nlayers])

    return output, b