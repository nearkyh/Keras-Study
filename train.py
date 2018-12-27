from __future__ import print_function

import os
import argparse

from utils.datasets import mnist
from utils.datasets import cifar10
from utils.models.cnn import ConvolutionNeuralNetwork


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mnist', type=str,
                    help="mnist, cifar10")
parser.add_argument('--batch_size', default=32, type=int,
                    help="Number of batch_size")
parser.add_argument('--epochs', default=10, type=int,
                    help="Number of epochs")
parser.add_argument('--gpu', default=False, type=bool,
                    help="Using GPU")
args = parser.parse_args()
batch_size = args.batch_size
epochs = args.epochs



if __name__ == '__main__':

    if args.dataset == 'mnist':
        mnist_get = mnist.pre_processing()
        num_classes = mnist_get['num_classes']
        x_train = mnist_get['x_train']
        x_test = mnist_get['x_test']
        y_train = mnist_get['y_train']
        y_test = mnist_get['y_test']
        input_shape = mnist_get['input_shape']

        cnn = ConvolutionNeuralNetwork(input_shape=input_shape,
                                       num_classes=num_classes,
                                       gpu=args.gpu)
        model = cnn.mnist_model()

        # 모델 학습
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))

        # 모델 평가
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        # 학습 모델 저장
        try:
            if not (os.path.isdir('utils/save_models')):
                os.makedirs(os.path.join('utils/save_models'))
        except Exception as e:
            print("Failed to create directory!!!")
        model.save('utils/save_models/cnn_mnist.h5')

    elif args.dataset == 'cifar10':
        cifar10_get = cifar10.pre_processing()
        num_classes = cifar10_get['num_classes']
        x_train = cifar10_get['x_train']
        x_test = cifar10_get['x_test']
        y_train = cifar10_get['y_train']
        y_test = cifar10_get['y_test']
        input_shape = cifar10_get['input_shape']

        cnn = ConvolutionNeuralNetwork(input_shape=input_shape,
                                       num_classes=num_classes)
        model = cnn.cifar10_model()

        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test, y_test))

        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        try:
            if not(os.path.isdir('utils/save_models')):
                os.makedirs(os.path.join('utils/save_models'))
        except Exception as e:
            print("Failed to create directory!!!")
        model.save('utils/save_models/cnn_cifar10.h5')
