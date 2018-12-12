import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers.convolutional import Conv2D
from keras.layers import Input, Add, Dense, Activation, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D, TimeDistributed
import numpy as np
import skimage.data
import selectivesearch
import os
import random
from random import shuffle
import tensorflow as tf

import FixedBatchNormalization
from mrcnn import model as modellib

ROI_SHAPE = (7,7)

class RoIPoolingConv(keras.engine.topology.Layer) :
    """
    ROI Pooling layer for 2D inputs
    pool_size : int
    pooling region의 크기. 7일 경우 7 x 7 크기의 출력

    num_rois : roi의 개수
    
    input_shape
    img(1, channels, rows, cols) or img(1, rows, clos, cahnnels)
    roi(1, rum_rois, 4) = (x, y, w, h)
    """

    def __init__(self, pool_size, num_rois, **kwargs) :
        self.dim_ordering = keras.backend.image_dim_ordering()
        
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoIPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape) :
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x, mask=None) :

        assert(len(x) == 2)

        img = x[0]
        rois = x[1]

        input_shape = keras.backend.shape(img)

        outputs = []

        for roi_idx in range(self.num_rois) :

            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            row_length = w / float(self.pool_size)
            col_length = h / float(self.pool_size)

            num_pool_regions = self.pool_size

            x = keras.backend.cast(x, 'int32')
            y = keras.backend.cast(y, 'int32')
            w = keras.backend.cast(w, 'int32')
            h = keras.backend.cast(h, 'int32')

            rs = tf.image.resize_images(img[:, y:y+h, x:x+w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = keras.backend.concatenate(outputs, axis=0)
        final_output = keras.backend.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        final_output = keras.backend.permute_dimensions(final_output, (0,1,2,3,4))

        return final_output

def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True) :

    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1,1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1,1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)

    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2), trainble=True) :

    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainble=trainble)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel, kernel_size), padding='same', name=conv_name_base + '2b', trainble=trainble)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainble=trainble)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainble=trainble)(input_tensor)
    shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

def NN_base(input_tesnor=None, trainable=False) :

    input_shape = (None, None, 3)

    if input_tensor is None :
        img_input = Input(shape=input_shape)

    else :
        if not keras.backend.is_keras_tensor(input_tesnor) :
            img_input = Input(tensor=input_tensor, shape=input_shape)

        else :
            img_input = input_tensor

    bn_axis = 3

    x = ZeroPadding2D((3, 3))(img_input)

    x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), trainable=trainable)
    ###12.12 여기까지함


class CNN :
    """
    이미지 전체를 입력으로 하는 CNN
    이미지의 feature vector를 출력
    """
    def __init__(self, input_shape, architecture) :
        input_image = Input(shape=input_shape)
        self.model = self.buildModel(input=input_image, architecture=architecture)

        return self.model

    def buildModel(self, input, architecture) :
        C1, C2, C3, C4, C5 = modellib.resnet_graph(input_image=input, architecture=architecture, stage5=True)
        CNNmodel = keras.models.Model(inputs=input, outputs=C5)
        CNNmodel.compile(optimizer='mean_squared_error', loss='sgd')

        return CNNmodel

    def feedforward(self, input) :
        return self.model.predict(x=input)
    

class FRCNN :
    def __init__(self, input_shape, output_class_num) :
        self.CNNmodel = CNN(input_shape, "resnet101")
        
        return None

    def poolRoI(self, input_feature_vector, RoI_tuple) :
        """
        C5의 출력 feature vector를 입력으로
        feature vector에서 RoI_tuple의 범위의 vector들을
        ROI_SHAPE로 나누어 max pooling하여 fixed length feature를 출력
        """

        x, y, h, w = RoI_tuple

