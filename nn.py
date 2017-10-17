import numpy as np

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




