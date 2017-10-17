import numpy as  np
import matplotlib.pyplot as plt
import pickle


# Function definitions

def GetGaussian(weightMatrix, randomDataPoints):
    w = weightMatrix
    x = randomDataPoints
    eucledianDistance = np.reshape(np.sum(np.square(w),axis=1),(1,-1)) + np.reshape(np.sum(np.square(x),axis=1),(-1,1)) - 2*np.dot(x,np.transpose(w))
    exponential = np.exp(-eucledianDistance/2)
    expSum = np.sum(exponential,axis = 1)
    gaussianNeuronMatrix = exponential/expSum[:,None]
    return gaussianNeuronMatrix


def FeedForward(x,weightMatrix,beta,outputType):
    b = np.dot(x,weightMatrix)
    output = np.tanh(beta*b)
    if(outputType == 1):
        gradient = beta*(1-(np.square(np.tanh(beta*b))))
        return output, gradient
    else:
        return output



# Data Intialization
# Read data
readFile = open("trainingDataInput.pickle","rb")
inputData = pickle.load(readFile)
readFile.close() 

readFile = open("trainingDataOutput.pickle","rb")
outputData = pickle.load(readFile)
readFile.close() 

# readFile = open("trainingDataDict.pickle","rb")
# trainingData = pickle.load(readFile)
# readFile.close() 

# Control parameters
nGaussianNeurons = 1000
nPatterns = np.size(inputData)
inputDimensions = 64*64
etaU = 0.02
nUpdatesU = 10**3
checkData = 10
batchSize = 100

# Unsupervised simple competetive learning

weightMatrixU = np.random.uniform(0,1,(nGaussianNeurons,inputDimensions))

flatten = lambda l: [item for sublist in l for item in sublist]

for nUnsupervised in range(0,nUpdatesU):
    j = np.random.random_integers(0,nPatterns-1,batchSize)
    flattenedInput = flatten(inputData[j])
    x = np.reshape(flattenedInput,(batchSize,inputDimensions))/255

    gaussianNeuronMatrix = GetGaussian(weightMatrixU,x)
    winningNeuronIndex = np.argmax(gaussianNeuronMatrix,axis=1)

    deltaW = etaU * (x-weightMatrixU[winningNeuronIndex])

    for i in range(0,batchSize):
        weightMatrixU[winningNeuronIndex[i]] = weightMatrixU[winningNeuronIndex[i]] + deltaW[i] 

    if(nUnsupervised%checkData == 0):
        print("n = ", nUnsupervised)


outputUnsupervised = np.empty((nPatterns,nGaussianNeurons))
for n in range(0,nPatterns,1000):
    print(n)
    j = np.arange(n, n+1000,1)
    flattenedInput = flatten(inputData[j])
    x = np.reshape(flattenedInput,(1000,inputDimensions))/255   
    outputUnsupervised[j,:] = GetGaussian(weightMatrixU,x)


# Control parameters

etaS = 0.1
beta = 0.5
epsilon = 0
nUpdatesS = 3*10**5
checkData = 100
batchSize = 100
outputClasses = 200

# Supervised simple percertron learning

bias = -1*np.ones((nPatterns,1))
inputSupervised = np.concatenate((outputUnsupervised,bias),axis = 1)
weightMatrixS = np.random.uniform(-1,1,(nGaussianNeurons + 1,outputClasses))
y = np.reshape(flatten(outputData),(nPatterns,outputClasses))

for nSupervised in range(0,nUpdatesS):
    j = np.random.random_integers(0,nPatterns-1,batchSize)
    xCurrent = inputSupervised[j]
    yCurrent = y[j]

    (output, gradient) = FeedForward(xCurrent,weightMatrixS,beta,1)

    delta = (yCurrent-output)*gradient
    deltaW = etaS*np.dot(np.transpose(xCurrent),delta)/batchSize
    
    weightMatrixS = weightMatrixS + deltaW

    if nSupervised%checkData == 0:
        testOutput = FeedForward(inputSupervised,weightMatrixS,beta,2)

        H = np.array((y-testOutput)**2).sum()/(2*nPatterns)

    if nSupervised%100 == 0:
        print("Energy = ",H)
        print("n sup = ",nSupervised)
        
    if nSupervised%checkData== 0:
        weightMatrixS = (1-epsilon)*weightMatrixS







