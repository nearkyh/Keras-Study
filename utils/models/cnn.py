import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils.training_utils import multi_gpu_model
from keras.backend import tensorflow_backend as tb


class ConvolutionNeuralNetwork:

    def __init__(self, input_shape, num_classes, gpu):
        self.input_shape = input_shape
        self.num_classes = num_classes
        if gpu == True:
            self.device = '/gpu:0'
        elif gpu == False:
            self.device = '/cpu:0'

    def mnist_model(self):
        with tb.tf.device(self.device):
            # CNN 모델 구성
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(3, 3),
                             activation='relu',
                             input_shape=self.input_shape))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.num_classes, activation='softmax'))
            # 멀티 GPU 할당
            # model = multi_gpu_model(model, gpus=2)
            model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])
            return model

    def cifar10_model(self):
        with tb.tf.device(self.device):
            model = Sequential()
            model.add(Conv2D(32, (3, 3), padding='same',
                             input_shape=self.input_shape))
            model.add(Activation('relu'))
            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(Activation('relu'))
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.num_classes))
            model.add(Activation('softmax'))

            # initiate RMSprop optimizer
            opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

            # Multi GPU
            # model = multi_gpu_model(model, gpus=2)

            # Let's train the model using RMSprop
            model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
            return model
