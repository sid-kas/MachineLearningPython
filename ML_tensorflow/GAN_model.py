import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

class SimpleGAN():

    def __init__(self,inputDimensions,outputDimensions):
        self.x_dim = inputDimensions
        self.y_dim = outputDimensions
        self.learning_rate = 0.0002
        self.xy=0; self.z=0
        self.d_loss=0; self.d_optim=0; self.D_output=0; self.d_sum=0 
        self.g_loss=0; self.g_optim=0; self.G_output=0; self.g_sum=0

    def discriminator(self,xy,activation='sigmoid',is_training=True,reuse=False):
        if activation == "tanh":
            g = tf.nn.tanh
        elif activation == "sigmoid":
            g = tf.nn.sigmoid
        with tf.variable_scope("discriminator", reuse=reuse):
            # define layers
            L1 = tf.layers.dense(inputs=xy, units= 32,activation=g,trainable=is_training,name="D_L1")
            L1 = tf.layers.dropout(L1,rate=0.3,training=is_training,name="D_L1_d")

            L2 = tf.layers.dense(inputs=L1, units= 64,activation=g,trainable=is_training,name="D_L2")
            L2 = tf.layers.dropout(L2,rate=0.5,training=is_training,name="D_L2_d")

            L3 = tf.layers.dense(inputs=L2, units= 32,activation=g,trainable=is_training,name="D_L3")
            L3 = tf.layers.dropout(L3,rate=0.3,training=is_training,name="D_L3_d")

            output = tf.layers.dense(inputs=L3,units= 1,activation=g,trainable=is_training,name="D_output")
            
            return output

    def generator_dense(self,z,activation = "tanh",is_training=True,reuse=False):
        if activation == "tanh":
            g = tf.nn.tanh
        elif activation == "sigmoid":
            g = tf.nn.sigmoid
        with tf.variable_scope("generator", reuse=reuse):
            # define layers
            L1 = tf.layers.dense(inputs=z, units= 64,activation=g,trainable=is_training,name="G_L1")
            L1 = tf.layers.dropout(L1,rate=0.3,training=is_training,name="G_L1_d")

            L2 = tf.layers.dense(inputs=L1, units= 64,activation=g,trainable=is_training,name="G_L2")
            L2 = tf.layers.dropout(L2,rate=0.4,training=is_training,name="G_L2_d")

            L3 = tf.layers.dense(inputs=L2, units= 32,activation=g,trainable=is_training,name="G_L3")
            L3 = tf.layers.dropout(L3,rate=0.3,training=is_training,name="G_L3_d")

            output = tf.layers.dense(inputs=L3,units=self.y_dim,activation=g,trainable=is_training,name="G_output")
            
            return output

    def generator_lstm(self,z,is_training=True,reuse=False):
        keep_prob = 0.4
        with tf.variable_scope("generator_lstm", reuse=reuse):
            # define layers
            dense_output = self.generator_dense(z,is_training=is_training,reuse=reuse)

            lstm_input = tf.expand_dims(dense_output,axis=0)
            lstm_cell = LSTMCell(32, forget_bias=0.3,use_peepholes=True)
            dropout_cell = DropoutWrapper(lstm_cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob, state_keep_prob=keep_prob)

            lstm_outputs, states = tf.nn.dynamic_rnn(dropout_cell, lstm_input, dtype=tf.float64,time_major=True,scope="G_rnn")

            output = tf.layers.dense(lstm_outputs[-1],units=self.y_dim,activation=tf.nn.sigmoid,trainable=is_training,name="G_output")
            
            return output
    

    def build_model(self):

        """ Placeholders """

        # real input data
        self.xy = tf.placeholder(dtype = tf.float64, shape = [None,self.x_dim+self.y_dim], name = "D_input_original")
        # Noise data
        self.z = tf.placeholder(dtype = tf.float64, shape = [None,self.x_dim], name = "G_input_noise") # Generator input
        
        """ Loss Function """

        # output of D for real data
        D_real_logits = self.discriminator(self.xy, is_training=True, reuse=False)

        # output of D for fake data
        G = self.generator_dense(self.z, is_training=True, reuse=False)
        zg = tf.concat([self.z,G],axis=1)
        D_fake_logits = self.discriminator(zg, is_training=True, reuse=True)

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))

        self.d_loss = d_loss_real + d_loss_fake

        # get loss for generator
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

        """ Training """

        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        
        d_vars = [var for var in t_vars if 'D_' in var.name]
        g_vars = [var for var in t_vars if 'G_' in var.name]

        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate) \
                      .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate*5) \
                      .minimize(self.g_loss, var_list=g_vars)

        """" Testing """
        # for test
        self.G_output = self.generator_dense(self.z, is_training=False, reuse=True)
        self.D_output =  self.discriminator(self.xy, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
    
    def train(self, xData,yData, updates = 10**3, batchSize = 100):
        session = tf.Session()
        nPatterns= session.run(tf.shape(yData)[0])
        # initialize all variables
        session.run(tf.global_variables_initializer())

        # saver to save model
        saver = tf.train.Saver()

        """ importance sampling """
        # kernel density
        x_kd = xData
        grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.01, 1.0, 500)}, cv=20) # 20-fold cross-validation
        grid.fit(x_kd)
        kde = grid.best_estimator_

        # summary writer
        writer = tf.summary.FileWriter('./logs/GAN_test', session.graph)
        counter = 0
        while(counter < updates):
            bootstrapIndx = np.random.random_integers(0,nPatterns-2,batchSize)
            xCurrent = xData[bootstrapIndx,:]
            yCurrent = yData[bootstrapIndx,:]
            batch_xy = np.concatenate((xCurrent,yCurrent),axis=1)
            # batch_z = np.random.uniform(0, 1, [batchSize, self.x_dim]).astype(np.float32)
            batch_z = kde.sample(n_samples=batchSize)
            t_vars = tf.trainable_variables()
            # update D network
            _, summary_str, d_loss = session.run([self.d_optim, self.d_sum, self.d_loss],
                                            feed_dict={self.xy:batch_xy, self.z: batch_z})
            writer.add_summary(summary_str, counter)

            # update G network
            _, summary_str, g_loss = session.run([self.g_optim, self.g_sum, self.g_loss], feed_dict={self.z: batch_z})
            writer.add_summary(summary_str, counter)

            # display training status
            if(counter%500==0):
                print(counter,'Updates completed out of', updates,', d_loss:  ',d_loss,', g_loss:', g_loss)
            counter += 1

        # save model
        saver.save(session,'./checkpoints/GAN_test')
        session.close()

    def test(self,xData):
        # graph inputs for visualize training results
        # sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))
        
        # Loading Global Variables
        init = tf.global_variables_initializer()
        session = tf.Session()  
        session.run(tf.global_variables_initializer())
        session.run(init)
        saver = tf.train.Saver()
        saver.restore(session,'./checkpoints/GAN_test')
        session.run(tf.global_variables())

        output = session.run(self.D_output, feed_dict={self.xy: xData})

        session.close()
        return output
