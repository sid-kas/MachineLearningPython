import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import pickle
from collections import deque # Adam Optimizer


def sample_test():
    networkConfig = json.load(open('Configuration.json'))
    inputColumns = [] # input column names as in the pd dataframe
    outputColumns = []
    columns = inputColumns + outputColumns
    data = [] # import data here
    training_data = pd.DataFrame(data, columns = columns)
    

    mlp = MLPClassifier(networkConfig)
    mlp.train(training_data, inputColumns, outputColumns)

class MLPClassifier():
    def __init__(self, networkConfig):
        self.hiddenUnit = []
        self.neuronOutput = []
        self.networkConfig = networkConfig
        self.weightMatrix = []
        if (networkConfig["configMLP"]["optimizer"] == "adam"):
            N = (networkConfig["configMLP"]["architecture"]["hiddenLayers"] + 1) * 2
            a = np.zeros(N)
            self.mT = deque(iterable = a, maxlen = N) # First Moment Vector - Adam Optimizer
            self.vT = deque(iterable = a, maxlen = N) # Second Moment Vector - Adam Optimizer
            self.timeStep = 1

    
    def train(self,trainingData ,inputColumns, outputColumns, weightsFileName=None): ### training data must be in pandas dataFrame format
        nHiddenLayers = self.networkConfig["configMLP"]["architecture"]["hiddenLayers"]
        updates = self.networkConfig["configMLP"]["updates"]
        batchSize = self.networkConfig["configMLP"]["batchSize"]
        checkData = self.networkConfig["configMLP"]["checkData"]
        checkDataAt = self.networkConfig["configMLP"]["checkDataAt"]
        
        if weightsFileName != None:
            fileName_weights = weightsFileName
        else:    
            fileName_weights = self.networkConfig["configMLP"]["weightMatrixFileName"]
        self.weightMatrix = InitializeWeightMatrix(len(inputColumns), len(outputColumns), self.networkConfig)

        # Data Normalization
        if (self.networkConfig["configMLP"]["dataNormalization"]):
            data = (trainingData - trainingData.mean()) / (trainingData.std())
        else:
            data = trainingData

        if (self.networkConfig["configMLP"]["dataShuffle"]):
            (x_train, x_valid, y_train, y_valid) = train_test_split(data[inputColumns], data[outputColumns], 
                                                                  test_size = self.networkConfig["configMLP"]["validationDataSize"])
        else:
             (x_train, x_valid, y_train, y_valid) = train_test_split(data[inputColumns], data[outputColumns], 
                                                                   test_size = self.networkConfig["configMLP"]["validationDataSize"],
                                                                   shuffle = False)

        # convert pandas dataFrame to numpy array
        x_train = x_train.values; y_train = y_train.values
        x_valid = np.transpose(x_valid.values); y_valid = np.transpose(y_valid.values)

        nPatterns = np.size(x_train,axis = 0)
        energyTraining = []
        n = 0
        # main loop
        while(n < updates):
            bootstrapIndx = np.random.random_integers(0,nPatterns-2,batchSize)
            # bootstrapIndx = np.arange(i*batchSize,(i+1)*batchSize)
            xCurrent = np.transpose(x_train[bootstrapIndx])
            yCurrent = np.transpose(y_train[bootstrapIndx])
            self.__feedForward(xCurrent)
            self.__optimizer(xCurrent,yCurrent)
            if (checkData and (n%checkDataAt == 0)):
                self.__feedForward(x_valid)
                validOutput = self.neuronOutput[nHiddenLayers]
                H = np.array((y_valid-validOutput)**2).sum()/(2*np.size(x_valid, axis=1))
                energyTraining.append(H)
                # save weightmatrix
                saveArray = open(fileName_weights,"wb")
                pickle.dump(self.weightMatrix, saveArray)
                saveArray.close()
                print(n,'Updates completed out of', updates,', Error: ',H)
            n += 1
            self.timeStep += 1
        print('Training completed with, Error: ',H,'; Updates: ', updates) 


    def predict(self,testData,inputColumns, outputColumns, weightsFileName = None,returnOutput = False):
        # Data Normalization
        if (self.networkConfig["configMLP"]["dataNormalization"]):
            data = (testData - testData.mean()) / (testData.std())
        else:
            data = testData
        x_test = data[inputColumns]; y_test = data[outputColumns]
        x = np.transpose(x_test.values)
        y  = np.transpose(y_test.values)
        nPatterns = len(x)
        weightMatrix = InitializeWeightMatrix(len(inputColumns), len(outputColumns), self.networkConfig)
        if weightsFileName != None:
            openFile = open(weightsFileName)
            weightMatrix = pickle.load(openFile)
            openFile.close()
            self.weightMatrix = weightMatrix
        else:
            self.weightMatrix = weightMatrix
        self.__feedForward(x)
        testOutput = self.neuronOutput.pop()
        H = np.array((y-testOutput)**2).sum()/(2*nPatterns)
        print ('Error: ',H)
        if returnOutput:
            return testOutput


    def __feedForward(self, xData):
        weightMatrix = self.weightMatrix
        nHiddenLayers = self.networkConfig["configMLP"]["architecture"]["hiddenLayers"]
        hiddenUnit = []; neuronOutput = [] # HiddenUnit - b, NeuronOutput - V
        g = self.networkConfig["configMLP"]["activationFunction"]
        # input to hidden layer
        hiddenUnit.append(np.dot(weightMatrix.input2Hidden["weights"], xData) 
                            - weightMatrix.input2Hidden["bias"])
        neuronOutput.append(self.__activationFunction(g,hiddenUnit[0]))

        # hidden to hidden
        for i in range(nHiddenLayers-1):
            hiddenUnit.append(np.dot(weightMatrix.hidden2Hidden[i]["weights"],neuronOutput[i])
                                - weightMatrix.hidden2Hidden[i]["bias"])
            neuronOutput.append(self.__activationFunction(g,hiddenUnit[i+1]))

        # hidden to output
        hiddenUnit.append(np.dot(weightMatrix.hidden2Output["weights"],neuronOutput[i+1])
                          - weightMatrix.hidden2Output["bias"])
        neuronOutput.append(self.__activationFunction(g, hiddenUnit[i+2]))

        self.hiddenUnit = hiddenUnit
        self.neuronOutput = neuronOutput
        
    def __optimizer(self,xData,yData): 
        # Vanilla Gradient descent optimizer 
        nHiddenLayers = self.networkConfig["configMLP"]["architecture"]["hiddenLayers"]
        g = self.networkConfig["configMLP"]["activationFunction"]
        gPrime = g + "_gradient"
        eta = self.networkConfig["configMLP"]["learningRate"]
        delta = deque() # error propagation

        # output to hidden
        activationGrad = self.__activationFunction(gPrime,self.hiddenUnit.pop())        
        delta.append(((yData - self.neuronOutput.pop())*activationGrad))

        # hidden to hidden
        activationGrad = self.__activationFunction(gPrime,self.hiddenUnit.pop())
        errorTimesW = np.dot(np.transpose(self.weightMatrix.hidden2Output["weights"]), delta[0])        
        delta.append(errorTimesW * activationGrad)

        backwardIndx = range(nHiddenLayers-2,-1,-1)
        for i in range(nHiddenLayers-1):
            iX = backwardIndx[i]
            i += 1
            activationGrad = self.__activationFunction(gPrime,self.hiddenUnit.pop())
            errorTimesW = np.dot(np.transpose(self.weightMatrix.hidden2Hidden[iX]["weights"]), delta[i])
            delta.append(errorTimesW * activationGrad)

        # Update weights
        if (self.networkConfig["configMLP"]["optimizer"] == "SGD"): # SGD - Stochastic Gradient Descent
            self.weightMatrix.hidden2Output["weights"] += eta * np.dot(delta[0],np.transpose(self.neuronOutput.pop()))
            self.weightMatrix.hidden2Output["bias"] += -eta * np.sum(delta[0], axis = 1, keepdims = True)
            backwardIndx = range(nHiddenLayers-2,-1,-1)
            for i in range(nHiddenLayers-1):
                iX = backwardIndx[i]
                self.weightMatrix.hidden2Hidden[iX]["weights"] += eta * np.dot(delta[i+1],np.transpose(self.neuronOutput.pop()))
                self.weightMatrix.hidden2Hidden[iX]["bias"] += -eta * np.sum(delta[i+1], axis = 1, keepdims = True)

            self.weightMatrix.input2Hidden["weights"] += eta * np.dot(delta[nHiddenLayers],np.transpose(xData))
            self.weightMatrix.input2Hidden["bias"] += -eta * np.sum(delta[nHiddenLayers], axis = 1, keepdims = True)
        
        elif (self.networkConfig["configMLP"]["optimizer"] == "adam"): # Adam Optimizer
            self.weightMatrix.hidden2Output["weights"] += self.__adamOptimizer(np.dot(delta[0],np.transpose(self.neuronOutput.pop())))
            self.weightMatrix.hidden2Output["bias"] += -self.__adamOptimizer(np.sum(delta[0], axis = 1, keepdims = True))
            backwardIndx = range(nHiddenLayers-2,-1,-1)
            for i in range(nHiddenLayers-1):
                iX = backwardIndx[i]
                self.weightMatrix.hidden2Hidden[iX]["weights"] += self.__adamOptimizer(np.dot(delta[i+1],np.transpose(self.neuronOutput.pop())))
                self.weightMatrix.hidden2Hidden[iX]["bias"] += -self.__adamOptimizer(np.sum(delta[i+1], axis = 1, keepdims = True))

            self.weightMatrix.input2Hidden["weights"] += self.__adamOptimizer(np.dot(delta[nHiddenLayers],np.transpose(xData)))
            self.weightMatrix.input2Hidden["bias"] += -self.__adamOptimizer(np.sum(delta[nHiddenLayers], axis = 1, keepdims = True))

    def __adamOptimizer(self, currentGradient):
        alpha = self.networkConfig["configMLP"]["adamOptimizer"]["alpha"] # Learning Rate
        beta1 = self.networkConfig["configMLP"]["adamOptimizer"]["beta1"]
        beta2 = self.networkConfig["configMLP"]["adamOptimizer"]["beta2"]
        epsilon = self.networkConfig["configMLP"]["adamOptimizer"]["epsilon"]
        t = self.timeStep
        mt = (beta1 * self.mT.popleft()) + (1 - beta1) * currentGradient
        vt = (beta2 * self.vT.popleft()) + (1 - beta2) * (np.square(currentGradient))
        mTPrime = mt / (1 - (beta1 ** t))
        vTPrime = vt / (1 - (beta2 ** t))
        gradient = (alpha * mTPrime) / (np.sqrt(vTPrime) + epsilon)
        self.mT.append(mt)
        self.vT.append(vt)
        return gradient

    def __activationFunction(self, g, data):
        beta = 0.5
        if g == "tanh":
            return np.tanh(beta*data,dtype=float)
        elif g == "tanh_gradient":
            return beta*(1-(np.square(np.tanh(beta*data,dtype=float))))
        elif g == "sigmoid":
            a = 0.5 * (1 + np.tanh(beta * data))
            return a
        elif g == "sigmoid_gradient":
            sigma = 0.5 * (1 + np.tanh(beta * data))
            return (sigma * (1 - sigma))

class InitializeWeightMatrix():
    def __init__(self, inputDimension, outputDimension, networkConfig):
        architecture = networkConfig["configMLP"]["architecture"]
        self.architecture = architecture
        (minWeight, maxWeight) = networkConfig["configMLP"]["weightsRange"]
        self.input2Hidden = {"weights": np.random.uniform(minWeight, maxWeight, 
                                        (architecture["hiddenUnits"][0], inputDimension)), 
                             "bias": np.random.uniform(minWeight, maxWeight, (architecture["hiddenUnits"][0], 1)) }
        hiddenLayersWeight = []
        for i in range(architecture["hiddenLayers"] - 1):
            hiddenLayersWeight.append({"weights": np.random.uniform(minWeight, maxWeight, (architecture["hiddenUnits"][i + 1], 
                                                  architecture["hiddenUnits"][i])), 
                                       "bias": np.random.uniform(minWeight, maxWeight, (architecture["hiddenUnits"][i + 1], 1))})
        self.hidden2Hidden = hiddenLayersWeight
        self.hidden2Output = {"weights": np.random.uniform(minWeight, maxWeight, 
                                         (outputDimension, architecture["hiddenUnits"][(architecture["hiddenLayers"] - 1)])), 
                              "bias": np.random.uniform(minWeight, maxWeight, (outputDimension, 1))}

