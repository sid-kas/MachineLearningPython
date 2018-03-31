import numpy as np
import matplotlib.pyplot as plt
import cv2
import urllib.request
import os
import pickle
import matplotlib.image as img
from os import listdir
from os.path import isfile, join




""" Plot functions """

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



""" Important math functions """

def EucledianDistance(x, weightMatrix):
    w = weightMatrix    
    term1 = np.reshape(np.sum(np.square(w),axis=1),(1,-1)) + np.reshape(np.sum(np.square(x),axis=1),(-1,1))
    # shapeW = np.shape(w)
    # wTranspose = np.reshape(flatten(zip(*w)),(shapeW[1],shapeW[0]))
    term2 = 2*np.dot(x,np.transpose(w))
    eucledianDistance = term1 - term2      
    return eucledianDistance

def GetGaussian(weightMatrix, randomDataPoints):
    w = weightMatrix
    x = randomDataPoints
    eucledianDistance = np.reshape(np.sum(np.square(w),axis=1),(1,-1)) + np.reshape(np.sum(np.square(x),axis=1),(-1,1)) - 2*np.dot(x,np.transpose(w))
    exponential = np.exp(-eucledianDistance/2)
    expSum = np.sum(exponential,axis = 1)
    gaussianNeuronMatrix = exponential/expSum[:,None]
    return gaussianNeuronMatrix

flatten = lambda l: [item for sublist in l for item in sublist]

""" Get raw images from URLs"""

def get_raw_images(link,directory):
    image_urls = urllib.request.urlopen(link).read().decode()

    if not os.path.exists(directory):
        os.makedirs(directory)
    pic_num = 1

    for i in image_urls.split('\n'):
        try:
            print(i)
            urllib.request.urlretrieve(i, directory + '/' + str(pic_num) + '.jpg')
            img = cv2.imread(directory+'/'+str(pic_num) + '.jpg', cv2.IMREAD_COLOR)
            resized_image = cv2.resize(img, (100,100))
            cv2.imwrite(directory+'/'+str(pic_num) + '.jpg', resized_image)
            pic_num += 1

        except Exception as e:
            print(str(e))

link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07687626'
directory = 'bread'
get_raw_images(link,directory)


""" Activation functions """

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

def softmax(x):
    x_exp = np.exp(np.array(x,dtype=float),dtype=float)
    x_exp_sum = np.sum(x_exp,axis = 1)
    return x_exp/x_exp_sum[:,None]

ReLU = lambda x: list(map(lambda x: x if x>0 else 0,x))
ReLU_Gradient = lambda x: list(map(lambda x: 1 if x>0 else 0,x))

LeakyReLU = lambda x,a: list(map(lambda x: x if x>0 else a*x,x))
LeakyReLU_GRadient = lambda x,a: list(map(lambda x: 1 if x>0 else a*1,x))

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

""" Random utilities"""

def img2Vec():
    raw = np.loadtxt('tiny-imagenet-200/wnids.txt',dtype=str)
    file_list = []
    for n in raw:
        text = n[2:]
        text = text[:9]
        file_list.append(text)

    matrix = np.identity(200)



    images1 = np.empty((500*200,1), dtype=object)
    images2 = np.empty(500*200, dtype=object)
    images3 = np.empty(500*200, dtype=object)
    nold = 0
    count = 0
    for file in raw:
        mypath='tiny-imagenet-200/train/'+ file + '/images/'
        onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
        print(file)
        out = matrix[count]
        for n in range(0, len(onlyfiles)):
            k = nold + n
            imgMatrix = cv2.imread( join(mypath,onlyfiles[n]), cv2.IMREAD_GRAYSCALE )
            inputVec = np.array(np.reshape(imgMatrix,(64*64,1))).flatten()
            dict = {'Name': str(file),'Input':inputVec,'Output':out}
            
            images1[k] = dict
            images2[k] = inputVec
            images3[k] = out

        nold = nold + n + 1
        count += 1

    #save data
    saveArray = open("trainingDataDict.pickle","wb")
    pickle.dump(images1, saveArray)
    saveArray.close()

    saveArray = open("trainingDataInput.pickle","wb")
    pickle.dump(images2, saveArray)
    saveArray.close()
    saveArray = open("trainingDataOutput.pickle","wb")
    pickle.dump(images3, saveArray)
    saveArray.close()

    #read data
    # f2 = open("trainingDataDict.pickle","rb")
    # dataread = pickle.load(f2)
    # f2.close() 










    