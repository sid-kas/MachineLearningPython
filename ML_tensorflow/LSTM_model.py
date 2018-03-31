import tensorflow as tf
import numpy as np


class SimpleLSTM():
    def __init__(self,inputDimensions,outputDimensions,scope):
        self.x_dim = inputDimensions
        self.y_dim = outputDimensions
        self.hiddenUnits = 100
        self.learningRate = 0.002
        with tf.variable_scope(scope):
            # Input
            self.x = tf.placeholder(shape=[None, self.x_dim], dtype=tf.float32)
            self.y = tf.placeholder(shape=[None, self.y_dim], dtype=tf.float32, name="y")
            # Recurrent network for temporal dependencies
            # LSTM cell initialization
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.hiddenUnits, state_is_tuple=True)
            # initial state from lstm cell
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.state_init = [c_init, h_init]
            # state placeholder
            self.c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            self.h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h], name="h")

            rnn_in = tf.expand_dims(self.x, [0])
            state_in = tf.contrib.rnn.LSTMStateTuple(self.c_in, self.h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in,
                initial_state=state_in,
                time_major=False,
                scope=scope)
            lstm_c, lstm_h = lstm_state
            self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
            rnn_out = tf.reshape(lstm_outputs, [-1, self.hiddenUnits])
            self.output = tf.layers.dense(inputs=rnn_out,units=self.y_dim,activation=tf.nn.tanh,name=scope)

            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(self.y,self.output))

            self.optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)

    def train(self,xData,yData,updates=10**3,timeSteps=5):
        session = tf.Session()
        nPatterns= session.run(tf.shape(yData)[0])
        # initialize all variables
        session.run(tf.global_variables_initializer())
        # saver
        saver = tf.train.Saver()

        rnn_state = self.state_init
        # main loop
        X,Y = self.preprocess(xData,yData,timeSteps)
        counter = 0
        n = 0
        while(counter < updates):
            bootstrapIndx = np.random.random_integers(0,nPatterns-2,10)
            xCurrent = xData[bootstrapIndx,:]
            yCurrent = yData[bootstrapIndx,:]
            
            _,nextState,error,_ =  session.run([self.output,self.state_out,self.loss,self.optimizer],
                                               feed_dict={self.x: xCurrent,
                                                          self.y: yCurrent,
                                                          self.c_in: rnn_state[0],
                                                          self.h_in: rnn_state[1]})
            
            rnn_state = nextState
    
            if(counter%500==0):
                print(counter,'Updates completed out of', updates,', loss:  ',error)

            if(n>=(nPatterns-timeSteps-2)):
                n=0
            n += 1
            counter += 1

        # save model
        saver.save(session,'./checkpoints/lstm_test')
        session.close()
        

    def test(self,xData,yData):
        nPatterns = len(xData)
        output_acc = np.zeros((nPatterns,self.y_dim),dtype=float)
        # Loading Global Variables
        init = tf.global_variables_initializer()
        session = tf.Session()  
        session.run(tf.global_variables_initializer())
        session.run(init)
        saver = tf.train.Saver()
        saver.restore(session,'./checkpoints/lstm_test')
        session.run(tf.global_variables())
        rnn_state = self.state_init
        loss = 0
        for i in range(nPatterns):
            xCurrent = xData[[i],:]
            yCurrent = yData[[i],:]
            l,output, nextState = session.run([self.loss,self.output,self.state_out], feed_dict={self.x: xCurrent,
                                                                                                self.y: yCurrent,
                                                                                                self.c_in: rnn_state[0],
                                                                                                    self.h_in: rnn_state[1]})
            rnn_state = nextState
            output_acc[i,:] = output
            loss+=l
        session.close()
        print("loss: ",loss/nPatterns)
        return output_acc
        

    def preprocess(self, dataX, dataY,timesteps):
        inputDimensions = np.shape(dataX)[1]
        outputDimensions = np.shape(dataY)[1]
        samples = len(dataX) - timesteps
        X = np.zeros([samples, timesteps, inputDimensions], dtype = np.float)
        Y = np.zeros([samples, outputDimensions], dtype = np.float)
        for i in range(samples):
            xIndex = range(i, (i+timesteps))
            yIndex = i+timesteps
            X[i,:,:] = dataX[xIndex]
            Y[i,:] = dataY[yIndex]

        return X, Y