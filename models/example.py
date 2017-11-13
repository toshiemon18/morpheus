# Example for model script

import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Lambda, Dropout, Reshape
from keras.layers import  MaxPooling2D, BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers import GRU
from keras.layers.merge import Add
from keras.layers import LeakyReLU
import tensorflow as tf
import math
import os, sys
sys.path.append(os.path.dirname(os.pardir))
from implements.complex_batchnormalization import ComplexBatchNormalization

class example:
    def __init__(self):
        self.model_name = "ExampleNet"
        self.optimizer  = "adam"
        self.loss       = "categorical_crossentropy"
        self.metrics    = ["accuracy"]

    # argument : shapes(n-dim)
    # shapes = (first-dim, second-dim, num-of-classes)
    def model(self, shapes):
        # === Params
        nfreqs, nduration, nclasses = shapes
        pool_size = (2, 2)
        stride_size = (2, 2)
        fs1 = 64
        fs2 = 128
        fs3 = 256
        fs4 = 512
        fs5 = 512
        activation = "relu"

        # === Input layer
        ir = Input(shape=(1, nfreqs, nduration), name='inputr')
        ij = Input(shape=(1, nfreqs, nduration), name='inputj')
        # === input normalization
        cbn = ComplexBatchNormalization(ir, ij, name="complex_norm")
        ir_ = cbn(ir)
        ij_ = cbn(ij)
        # === Log-scale amplitude spectrum
        sqr = Lambda(lambda x: x**2)(ir)
        sqj = Lambda(lambda x: x**2)(ij)
        sqrj = Add()([sqr, sqj])
        inputs = Lambda(lambda x: x**0.5, name="absolute")(sqrj)
        # inputs = Lambda(lambda x: tf.div(tf.log(x), tf.log(tf.constant(10, dtype=tf.float32))), name="log-amp_spectrum")(inputs)
        # === ConvBlock1
        x = Conv2D(fs1, kernel_size=(3, 3), padding='same', activation="relu", use_bias=True, name="conv11", data_format="channels_first")(inputs)
        x = Conv2D(fs1, kernel_size=(3, 3), padding='same', activation="relu", use_bias=True, name="conv12", data_format="channels_first")(inputs)
        x = MaxPooling2D(pool_size=pool_size, strides=stride_size, padding='valid', data_format="channels_first", name="max_pool1")(x)
        # === ConvBlock2
        x = Conv2D(fs2, kernel_size=(3, 3), padding='same', activation="relu", use_bias=True, name="conv21", data_format="channels_first")(x)
        x = Conv2D(fs2, kernel_size=(3, 3), padding='same', activation="relu", use_bias=True, name="conv22", data_format="channels_first")(x)
        x = MaxPooling2D(pool_size=pool_size, strides=stride_size, padding='valid', data_format="channels_first", name="max_pool2")(x)
        # === ConvBlock3
        x = Conv2D(fs3, kernel_size=(3, 3), padding='same', activation="relu", use_bias=True, name="conv31", data_format="channels_first")(x)
        x = Conv2D(fs3, kernel_size=(3, 3), padding='same', activation="relu", use_bias=True, name="conv32", data_format="channels_first")(x)
        x = Conv2D(fs3, kernel_size=(3, 3), padding='same', activation="relu", use_bias=True, name="conv33", data_format="channels_first")(x)
        x = MaxPooling2D(pool_size=pool_size, strides=stride_size, padding='valid', data_format="channels_first", name="max_pool3")(x)
        # === ConvBlock4
        x = Conv2D(fs4, kernel_size=(3, 3), padding='same', activation="relu", use_bias=True, name="conv41", data_format="channels_first")(x)
        x = Conv2D(fs4, kernel_size=(3, 3), padding='same', activation="relu", use_bias=True, name="conv42", data_format="channels_first")(x)
        x = Conv2D(fs4, kernel_size=(3, 3), padding='same', activation="relu", use_bias=True, name="conv43", data_format="channels_first")(x)
        x = MaxPooling2D(pool_size=pool_size, strides=stride_size, padding='valid', data_format="channels_first", name="max_pool4")(x)
        # === ConvBlock5
        x = Conv2D(fs5, kernel_size=(3, 3), padding='same', activation="relu", use_bias=True, name="conv51", data_format="channels_first")(x)
        x = Conv2D(fs5, kernel_size=(3, 3), padding='same', activation="relu", use_bias=True, name="conv52", data_format="channels_first")(x)
        x = Conv2D(fs5, kernel_size=(3, 3), padding='same', activation="relu", use_bias=True, name="conv53", data_format="channels_first")(x)
        x = MaxPooling2D(pool_size=pool_size, strides=stride_size, padding='valid', data_format="channels_first", name="max_pool5")(x)
        # === Recognition block
        # === Flatten - FullConnect*3 - Softmax
        x = Flatten()(x)
        # x = Dense(1024, name="fc1", activation="relu")(x)
        # x = Dropout(0.5)(x)
        # x = Dense(1024, name="fc2", activation="relu")(x)
        # x = Dropout(0.5)(x)
        x = Dense(nclasses, name="full_c")(x)
        output = Activation('softmax', name="softmax")(x)
        model = Model(inputs=[ir, ij], outputs=[output])
        return model

    # argument : None
    # return compile options, optimizer, loss function, and metrics list
    def compile_options(self):
        return (self.optimizer, self.loss, self.metrics)

