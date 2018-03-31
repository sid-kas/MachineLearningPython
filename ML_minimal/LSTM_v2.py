####################################
## LSTM WITH PEEPHOLE CONNECTIONS ##
####################################

import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import time
import copy
from collections import deque

def sample_test():
    networkConfig = json.load(open('Configuration.json'))
    inputColumns = [] # input column names as in the pd dataframe
    outputColumns = []
    columns = inputColumns + outputColumns
    data = [] # import data here
    training_data = pd.DataFrame(data, columns = columns)
    

    lstm = lstm_network(networkConfig)
    lstm.train(training_data, inputColumns, outputColumns)

class lstm_network():
    def __init__(self,networkConfig):
        self.networkConfig = networkConfig

    def train(self,trainingData ,inputColumns, outputColumns):
        # --| Control parameters
        updates = self.networkConfig["configLSTM"]["updates"]
        batchSize = self.networkConfig["configLSTM"]["batchSize"]
        windowSize = self.networkConfig["configLSTM"]["windowSize"]
        checkData = self.networkConfig["configLSTM"]["checkData"]

        # --| Data Normalization
        if (self.networkConfig["configLSTM"]["dataNormalization"]):
            # data = (trainingData - trainingData.mean()) / (trainingData.std())
            data = (trainingData[inputColumns] - trainingData[inputColumns].mean()) / (trainingData[inputColumns].std())
        else:
            data = trainingData
        print("data normalized")
        if (self.networkConfig["configLSTM"]["dataShuffle"]):
            (x_train, x_test, y_train, y_test) = train_test_split(data[inputColumns], data[outputColumns], 
                                                                  test_size = self.networkConfig["configLSTM"]["testDataSize"])
        else:
             (x_train, x_test, y_train, y_test) = train_test_split(data[inputColumns], data[outputColumns], 
                                                                   test_size = self.networkConfig["configLSTM"]["testDataSize"],
                                                                   shuffle = False)
        print("train_test_split done!")
        # --| Convert pandas dataFrame to numpy array
        x_train = x_train.values; y_train = y_train.values
        x_test = np.transpose(x_test.values); y_test = np.transpose(y_test.values)

        # --| Network architecture initialization
        N = self.networkConfig["configLSTM"]["architecture"]["hiddenUnits"]
        M = len(inputColumns)
        K = len(outputColumns)
        # layer1
        w1 = lstm_layer_weights(M,N,self.networkConfig)
        s1 = lstm_states(N)
        p1 = lstm_partials(N)
        layer1 = lstm_layer( w1, s1, p1)
        # output layer
        w2 = np.random.uniform(-0.1,0.1,(K,N))
        outputLayer = output_layer(w2)

        ##### INTIALIZATION FOR TEST
        # layer1
        tests1 = lstm_states(N)
        testp1 = lstm_partials(N)
        testlayer1 = lstm_layer( w1, tests1, testp1)
        # output layer
        testoutputLayer = output_layer(w2)
        #####

        print("started training!!")
        # --| Main loop
        nPatterns = np.size(x_train,axis = 0)
        energyTraining = []; n = 0; nUpdates = 1
        # states = [[] for i in range(batchSize)]
        states = deque(iterable = [[] for i in range(windowSize+1)], maxlen =windowSize+1)
        s0 = lstm_states(N)
        states.append((s0, testoutputLayer))
        while(nUpdates < updates):
            bootstrapIndx = np.arange(n*batchSize*windowSize,(n+1)*batchSize*windowSize)
            xCurrent = np.array(x_train[bootstrapIndx])
            yCurrent = np.array(y_train[bootstrapIndx+1])

            # currentWindowSize = len(xCurrent)
            layer1.weights.windowSize = windowSize
            layer1.weights.timeStep = nUpdates
            # Forward pass
            for i in range(windowSize):
                x = np.transpose(xCurrent[range(i*batchSize, (i+1)*batchSize)])
                layer1.forwardPass(x)
                outputLayer.forwardPass(layer1.states.h)
                states.append((copy.deepcopy(layer1.states),copy.deepcopy(outputLayer)))
            
            endState = copy.deepcopy(layer1.states)
            # Backward pass
            for i in range(windowSize-1,-1,-1):
                x = xCurrent[range(i*batchSize, (i+1)*batchSize)]
                y = np.transpose(yCurrent[range(i*batchSize, (i+1)*batchSize)])
                currentStates = states.pop()
                outputLayer.y = currentStates[1].y ; outputLayer.y_ = currentStates[1].y_
                dh = outputLayer.backwardPass(currentStates[0].h,y)
                oldCellState = states[len(states)-1][0].c
                layer1.states = currentStates[0]
                layer1.backwardPass(x,dh, oldCellState)
            
            states[0] = (endState, testoutputLayer)

            if (checkData and (nUpdates%20 == 0)):
                testlayer1.forwardPass(x_test)
                testoutputLayer.forwardPass(testlayer1.states.h)
                testOutput = testoutputLayer.y
                H = error("squaredError", testOutput, y_test)
                energyTraining.append(H)
                # save weightmatrix
                saveArray = open("weightMatrix1LSTMTestData.pickle","wb")
                pickle.dump(w1, saveArray)
                saveArray.close()
                saveArray = open("weightMatrix2LSTMTestData.pickle","wb")
                pickle.dump(w2, saveArray)
                saveArray.close()
                print(nUpdates,'Updates completed out of', updates,', Energy: ',H)

            # if(n == (len(x_train)/batchSize*windowSize)-1):
            #     n = 0
            if n == 40:
                n = 0

            nUpdates += 1
            n += 1
            
    def validate(self):
        pass

class lstm_layer():
    def __init__(self, weightmatrix, states, partials):
        self.weights = weightmatrix
        self.states = states
        self.partials = partials

    def forwardPass(self, x):
        # x -> M x batchSize(1)
        # Block input
        self.states.a_ = np.dot(self.weights.Wa,x) + np.dot(self.weights.Ra,self.states.h) + self.weights.Ba
        self.states.a = activation("tanh",self.states.a_)
        # Input gate
        self.states.i_ = (np.dot(self.weights.Wi,x) + np.dot(self.weights.Ri,self.states.h) 
                          + (self.weights.Pi * self.states.c) + self.weights.Bi)
        self.states.i = activation("sigmoid",self.states.i_)
        # Forget gate
        self.states.f_ = (np.dot(self.weights.Wf,x) + np.dot(self.weights.Rf,self.states.h)
                          + (self.weights.Pf * self.states.c) + self.weights.Bf)
        self.states.f = activation("sigmoid",self.states.f_)
        # Cell state(CEC)
        self.states.c = self.states.i * self.states.a + self.states.f + self.states.c
        # Output gate
        self.states.o_ = (np.dot(self.weights.Wo,x) + np.dot(self.weights.Ro,self.states.h) 
                          + (self.weights.Po * self.states.c) + self.weights.Bo)
        self.states.o = activation("sigmoid",self.states.o_)
        # Block output
        self.states.h = self.states.o * activation("tanh", self.states.c)

    def backwardPass(self, x, dh, oldCellState):
        oldPartials = copy.deepcopy(self.partials) # Partials of t+1
        #val = (np.dot(self.weights.Ra, self.partials.da) + np.dot(self.weights.Ri, self.partials.di)
               #+ np.add(np.dot(self.weights.Rf, self.partials.df), np.dot(self.weights.Ro, self.partials.do), dtype=np.float_))
        self.partials.dh = dh #+ val 
        
        self.partials.do = np.multiply(self.partials.dh, activation("tanh",self.states.c), dtype= np.float_)
        self.partials.do_ = np.multiply(self.partials.df, activation("sigmoid_gradient",self.states.o_),dtype= np.float_ )

        self.partials.dc = (self.partials.dh * self.states.o * activation("tanh_gradient",self.states.c)
                            + self.weights.Po * self.partials.do_ + self.weights.Pi * self.partials.di_ 
                            + np.multiply(self.weights.Pf, self.partials.df_, dtype= np.float_) 
                            + np.multiply(self.partials.dc, self.states.f, dtype= np.float_))

        self.partials.df = np.multiply(self.partials.dc, oldCellState, dtype= np.float_)
        self.partials.df_ = np.multiply(self.partials.df, activation("sigmoid_gradient",self.states.f_),dtype= np.float_ )

        self.partials.di = np.multiply(self.partials.dc, self.states.a, dtype= np.float_)
        self.partials.di_ = np.multiply(self.partials.di, activation("sigmoid_gradient",self.states.i_),dtype= np.float_ )

        self.partials.da = np.multiply(self.partials.dc, self.states.i, dtype= np.float_)
        self.partials.da_ = np.multiply(self.partials.da, activation("tanh_gradient",self.states.a_),dtype= np.float_ )

        self.weights.dWa = np.dot(self.partials.da_,x); self.weights.dWi = np.dot(self.partials.di_,x)
        self.weights.dWf = np.dot(self.partials.df_,x); self.weights.dWo = np.dot(self.partials.do_,x)
        hT = np.transpose(self.states.h)
        self.weights.dRa = np.dot(self.partials.da_,hT); self.weights.dRi = np.dot(self.partials.di_,hT)
        self.weights.dRf = np.dot(self.partials.df_,hT); self.weights.dRo = np.dot(self.partials.do_,hT)

        self.weights.dPi = np.sum((self.states.c * oldPartials.di_), axis = 1, keepdims = True)
        self.weights.dPf = np.sum((self.states.c * oldPartials.df_), axis = 1, keepdims = True)
        self.weights.dPo = np.sum((self.states.c * self.partials.do_), axis = 1, keepdims = True)

        self.weights.dBa = np.sum(self.states.a_, axis = 1, keepdims = True)
        self.weights.dBi = np.sum(self.states.i_ , axis = 1, keepdims = True)
        self.weights.dBf = np.sum(self.states.f_, axis = 1, keepdims = True)
        self.weights.dBo = np.sum(self.states.o_, axis = 1, keepdims = True)

        # If there is another network below LSTM
        if False:
            dx = (np.dot(self.weights.Wa, self.partials.da_) + np.dot(self.weights.Wi, self.partials.di_)
                  + np.dot(self.weights.Wf, self.partials.df_) + np.dot(self.weights.Wo, self.partials.do_))
            self.weights.updateWeights()
            return dx
        else:
            self.weights.updateWeights()

      

class output_layer():
    def __init__(self, weightmatrix):
        self.Wy = weightmatrix
        self.y = []
        self.y_ = []
    def forwardPass(self,h):
        self.y_ = np.dot(self.Wy, h)
        self.y = activation("sigmoid",self.y_)

    def backwardPass(self,h,targetOutput):
        dy = error("squaredError_gradient",self.y,targetOutput ) * activation("sigmoid_gradient", self.y_) 
        dWy = np.dot(dy, np.transpose(h))
        dh = np.dot(np.transpose(self.Wy),dy)
        self.Wy -= 0.0001*dWy
        return dh

def activation(g, data):
    beta = 0.5
    if g == "tanh":
        return np.tanh(beta * data, dtype = np.float_)
    elif g == "tanh_gradient":
        return beta*(1-(np.square(np.tanh(beta*data,dtype=np.float_))))
    elif g == "sigmoid":
        # a = 1 + np.exp(-(beta * data),dtype = np.float_)
        # return np.divide(1, a, dtype = np.float_)
        a = 0.5 * (1 + np.tanh(beta * data))
        return a
    elif g == "sigmoid_gradient":
        # a = 1 + np.exp(-(beta * data),dtype = np.float_)
        # sigma = np.divide(1, a, dtype = np.float_)
        sigma = a = 0.5 * (1 + np.tanh(beta * data))
        return (sigma * (1 - sigma))
    elif g == "relu":
        a = 0.01
        x = np.array(data, dtype = np.float_)
        x[x < 0] = x[x < 0] * a
        return x
    elif g == "relu_gradient":
        a = 0.01
        x = np.array(data, dtype = np.float_)
        x[x > 0] = 1
        x[x < 0] = a
        return x

def error(f, output, targetOutput):
    if (f == "squaredError"):
        # E = np.array((targetOutput-output)**2).sum()/(2*np.size(output, axis=1))
        a = np.square(np.subtract(targetOutput, output, dtype = np.float_), dtype = np.float_)
        b = np.nansum(a, axis = 0, dtype = np.float_)
        c = np.nansum(b, dtype = np.float_)
        E = np.divide(c, (2*np.size(targetOutput, axis = 1)), dtype = np.float_)
        return E
    elif (f == "squaredError_gradient"):
        dE = -1 * np.subtract(targetOutput , output, dtype= np.float_)
        return dE


class lstm_layer_weights():
    def __init__(self,M,N,networkConfig):
        (minWeight, maxWeight) = networkConfig["configLSTM"]["weightsRange"]
        self.networkConfig = networkConfig
        # regular weights
        self.Wa = np.random.uniform(minWeight,maxWeight,(N,M))
        self.Wi = np.random.uniform(minWeight,maxWeight,(N,M))
        self.Wf = np.random.uniform(0.3,maxWeight,(N,M))
        self.Wo = np.random.uniform(minWeight,maxWeight,(N,M))
        # recurrent weights
        self.Ra = np.random.uniform(minWeight,maxWeight,(N,N))
        self.Ri = np.random.uniform(minWeight,maxWeight,(N,N))
        self.Rf = np.random.uniform(minWeight,maxWeight,(N,N))
        self.Ro = np.random.uniform(minWeight,maxWeight,(N,N))
        # peephole weights
        self.Pi = np.random.uniform(minWeight,maxWeight,(N,1))
        self.Pf = np.random.uniform(minWeight,maxWeight,(N,1))
        self.Po = np.random.uniform(minWeight,maxWeight,(N,1))
        # bias weights
        self.Ba = np.random.uniform(minWeight,maxWeight,(N,1))
        self.Bi = np.random.uniform(minWeight,maxWeight,(N,1))
        self.Bf = np.random.uniform(minWeight,maxWeight,(N,1))
        self.Bo = np.random.uniform(minWeight,maxWeight,(N,1))
        # derivatives of the loss function w.r.t weights
        self.dWa = np.random.uniform(minWeight,maxWeight,(N,M))
        self.dWi = np.random.uniform(minWeight,maxWeight,(N,M))
        self.dWf = np.random.uniform(minWeight,maxWeight,(N,M))
        self.dWo = np.random.uniform(minWeight,maxWeight,(N,M))
        self.dRa = np.random.uniform(minWeight,maxWeight,(N,N))
        self.dRi = np.random.uniform(minWeight,maxWeight,(N,N))
        self.dRf = np.random.uniform(minWeight,maxWeight,(N,N))
        self.dRo = np.random.uniform(minWeight,maxWeight,(N,N))
        self.dPi = np.random.uniform(minWeight,maxWeight,(N,1))
        self.dPf = np.random.uniform(minWeight,maxWeight,(N,1))
        self.dPo = np.random.uniform(minWeight,maxWeight,(N,1))
        self.dBa = np.random.uniform(minWeight,maxWeight,(N,1))
        self.dBi = np.random.uniform(minWeight,maxWeight,(N,1))
        self.dBf = np.random.uniform(minWeight,maxWeight,(N,1))
        self.dBo = np.random.uniform(minWeight,maxWeight,(N,1))
        self.windowSize = 1
        self.timeStep = 1
        if (networkConfig["configLSTM"]["optimizer"] == "adam"):
            N = 15* self.windowSize # update this 
            a = np.zeros(N) 
            self.mT = deque(iterable = a, maxlen = N) # First Moment Vector - Adam Optimizer
            self.vT = deque(iterable = a, maxlen = N) # Second Moment Vector - Adam Optimizer

    def updateWeights(self):
        o = self.networkConfig["configLSTM"]["optimizer"]
        if (o == "SGD"):
            optimizer = lambda x: x*0.0001
        elif (o == "adam"):
            optimizer = lambda x: self.__adamOptimizer(x)

        # regular weights
        self.Wa -= optimizer(self.dWa) 
        self.Wi -= optimizer(self.dWi)
        self.Wf -= optimizer(self.dWf)
        self.Wo -= optimizer(self.dWo)
        # recurrent weights
        self.Ra -= optimizer(self.dRa)
        self.Ri -= optimizer(self.dRi)
        self.Rf -= optimizer(self.dRf)
        self.Ro -= optimizer(self.dRo)
        # peephole weights
        self.Pi -= optimizer(self.dPi)
        self.Pf -= optimizer(self.dPf)
        self.Po -= optimizer(self.dPo)
        # bias weights
        self.Ba -= optimizer(self.dBa)
        self.Bi -= optimizer(self.dBi)
        self.Bf -= optimizer(self.dBf)
        self.Bo -= optimizer(self.dBo)
    
    def __adamOptimizer(self, currentGradient):
        alpha = self.networkConfig["configLSTM"]["adamOptimizer"]["alpha"] # Learning Rate
        beta1 = self.networkConfig["configLSTM"]["adamOptimizer"]["beta1"]
        beta2 = self.networkConfig["configLSTM"]["adamOptimizer"]["beta2"]
        epsilon = self.networkConfig["configLSTM"]["adamOptimizer"]["epsilon"]
        t = self.timeStep # a bit ambiguous -- but mostly global time step per 1 cycle
        mt = (beta1 * self.mT.popleft()) + (1 - beta1) * currentGradient
        vt = (beta2 * self.vT.popleft()) + (1 - beta2) * (currentGradient**2)
        b = np.float(1 - (beta1 ** t))
        mTPrime = np.divide(mt, b, dtype=float)
        vTPrime = np.divide(vt, b, dtype=float)
        gradient = np.divide((alpha * mTPrime), (np.sqrt(vTPrime) + epsilon), dtype=np.float_)
        self.mT.append(mt)
        self.vT.append(vt)
        return gradient

class lstm_states():
    def __init__(self,N):    
        self.a_ = np.zeros((N,1))
        self.a = np.zeros((N,1))
        self.i_ = np.zeros((N,1))
        self.i = np.zeros((N,1))
        self.f_ = np.zeros((N,1))
        self.f = np.zeros((N,1))
        self.c = np.zeros((N,1))
        self.o_ = np.zeros((N,1))
        self.o = np.zeros((N,1))
        self.h = np.zeros((N,1))

class lstm_partials():
    def __init__(self,N):    
        # partials
        self.da = np.zeros((N,1))
        self.di = np.zeros((N,1))
        self.df = np.zeros((N,1))
        self.da_ = np.zeros((N,1))
        self.di_ = np.zeros((N,1))
        self.df_ = np.zeros((N,1))
        self.dc = np.zeros((N,1))
        self.do = np.zeros((N,1))
        self.do_ = np.zeros((N,1))
        self.dh = np.zeros((N,1))


if __name__ == '__main__':
	main()