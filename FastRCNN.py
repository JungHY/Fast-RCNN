import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
import numpy as np
import skimage.data
import selectivesearch
import os
import random
from random import shuffle

from mrcnn import model as modellib

class FRCNN :
    def __init__(self, input_shape, output_class_num) :

        self.input_shape = input_shape
        self.input_image = Input(shape=self.input_shape)

        
        return None

    def buildCNN(self, input, arhitecture) :
        
        C1, C2, C3, C4, C5 = modelib.resnet_graph(input_image=input, arhitecture=arhitecture, stage5=True)

        self.CNNmodel = keras.models.Model(inputs=self.input_image, outputs=C5)
        self.CNNmodel.compile(optimizer='mean_squared_error', loss='sgd')

        





