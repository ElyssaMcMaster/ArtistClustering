import random
import os
import shutil
import random
import sys
from PIL import Image
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rotate
#from skimage.util import random_noise
from scipy import ndimage
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical  
from keras.layers import LeakyReLU 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,Input,Conv2D,MaxPooling2D,BatchNormalization,Softmax
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras import optimizers
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import cv2
#from main import manuscriptModel

def runModel(X_train, X_test, y_train, y_test):
        #resultsFile = os.mkdir('/home/researchstudent/Research/ARTH_Classification/ARTH493-Code/Results/ReportResultsWithSupp.txt')
        for item in range(len(X_train)):
                if X_train[item].shape != (250,250,3):
                 X_train[item] = cv2.cvtColor(X_train[item], cv2.COLOR_BGRA2BGR)
        #x = np.array(X_train[0])
        #print(x.shape)
        X_train = np.array(X_train)

        y_train = np.array(y_train)

        print(X_train.shape) # (3642, 250, 250, 3)
        print(y_train.shape) # (3642,)
        #print(X_test.shape) #this is a list
        #print(y_test.shape)

        y_train_1hot = to_categorical(y_train, 5) # (3642, 5)
        y_test_1hot = to_categorical(y_test, 5)      # (402, 5)

        #print(y_train.shape)
        #print(y_test.shape)

        CNN = Sequential()
        
        ## Add a convolutional layer with 32 filters, 3x3 kernel, relu activation, he uniform kernel initializer, same padding and input shape
        input_shape = (250,250,3)
        ## Add a convolutional layer with 32 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding 
        CNN.add(Conv2D(32,3,activation = 'relu',kernel_initializer = 'he_uniform', padding = 'same'))

        ## Add a batch normalization layer
        CNN.add(BatchNormalization())

        ## Add a max pooling 2d layer with 2x2 size
        CNN.add(MaxPool2D(pool_size = (2,2)))
        
        ## Add dropout layer of 0.1
        CNN.add(Dropout(0.1))
        
        ## Add a convolutional layer with 64 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
        CNN.add(Conv2D(64,3,activation = 'relu',kernel_initializer = 'he_uniform', padding = 'same'))
        
        ## Add a batch normalization layer
        CNN.add(BatchNormalization())
        
        ## Add a convolutional layer with 64 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
        CNN.add(Conv2D(64,3,activation = 'relu',kernel_initializer = 'he_uniform', padding = 'same'))
        
        ## Add a batch normalization layer
        CNN.add(BatchNormalization())
        
        ## Add a max pooling 2d layer with 2x2 size
        CNN.add(MaxPool2D(pool_size = (2,2)))
        
        ## Add dropout layer of 0.1
        CNN.add(Dropout(0.1))
        
        ## Add a convolutional layer with 128 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
        CNN.add(Conv2D(128,3,activation = 'relu',kernel_initializer = 'he_uniform', padding = 'same'))
        
        ## Add a batch normalization layer
        CNN.add(BatchNormalization())
        
        ## Add a convolutional layer with 128 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
        CNN.add(Conv2D(128,3,activation = 'relu',kernel_initializer = 'he_uniform', padding = 'same'))
        
        ## Add a batch normalization layer
        CNN.add(BatchNormalization())
        
        ## Add a max pooling 2d layer with 2x2 size
        CNN.add(MaxPool2D(pool_size = (2,2)))
        
        ## Add dropout layer of 0.1
        CNN.add(Dropout(0.1))
        
        ## Add a convolutional layer with 256 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
        CNN.add(Conv2D(256,3,activation = 'relu',kernel_initializer = 'he_uniform', padding = 'same'))
        
        ## Add a batch normalization layer
        CNN.add(BatchNormalization())
        
        ## Add a convolutional layer with 256 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
        CNN.add(Conv2D(256,3,activation = 'relu',kernel_initializer = 'he_uniform', padding = 'same'))
        
        ## Add a batch normalization layer
        CNN.add(BatchNormalization())
        
        ## Add a max pooling 2d layer with 2x2 size
        CNN.add(MaxPool2D(pool_size = (2,2)))
        
        ## Add dropout layer of 0.1
        CNN.add(Dropout(0.1))
        
        ## Add a convolutional layer with 256 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
        CNN.add(Conv2D(512,3,activation = 'relu',kernel_initializer = 'he_uniform', padding = 'same'))
        
        ## Add a batch normalization layer
        CNN.add(BatchNormalization())
        
        ## Add a convolutional layer with 256 filters, 3x3 kernel, relu activation, he uniform kernel initializer, and same padding
        CNN.add(Conv2D(512,3,activation = 'relu',kernel_initializer = 'he_uniform', padding = 'same'))
        
        ## Add a batch normalization layer
        CNN.add(BatchNormalization())
        
        ## Add a max pooling 2d layer with 2x2 size
        CNN.add(MaxPool2D(pool_size = (2,2)))
        
        ## Add dropout layer of 0.1
        CNN.add(Dropout(0.1))
        
        ## Flatten the resulting data
        CNN.add(Flatten())
        
        ## Add a dense layer with 128 nodes, relu activation and he uniform kernel initializer
        CNN.add(Dense(units = 512, activation = "relu",kernel_initializer="he_normal"))
        
        ## Add a batch normalization layer
        CNN.add(BatchNormalization())
        
        ## Add dropout layer of 0.2
        CNN.add(Dropout(0.2))
        
        ## Add a dense softmax layer
        CNN.add(Dense(units = 5, activation= "softmax"))
        
        ## Set up early stop training with a patience of 3
        EarlyStop = EarlyStopping(patience = 3)
        
        ## Compile the model with adam optimizer, categorical cross entropy and accuracy metrics
        CNN.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics= ['accuracy'])

        history = History()
        
        ## Fit the model with the generated data, 200 epochs, steps per epoch and validation data defined. 
        CNN.fit(X_train, y_train_1hot, epochs=100, callbacks=[history, EarlyStop], validation_split=0.10)

        CNN.summary()
        
        ## Save model for later use
        CNN.save('./CNN_Model_Std')