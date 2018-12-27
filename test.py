import numpy as np
import matplotlib.pyplot as plt

from utils.datasets import mnist
from utils.datasets import cifar10
from keras.models import load_model


# mnist_get = mnist.pre_processing()
# num_classes = mnist_get['num_classes']
# x_train = mnist_get['x_train']
# x_test = mnist_get['x_test']
# y_train = mnist_get['y_train']
# y_test = mnist_get['y_test']
# input_shape = mnist_get['input_shape']

cifar10_get = cifar10.pre_processing()
num_classes = cifar10_get['num_classes']
x_train = cifar10_get['x_train']
x_test = cifar10_get['x_test']
y_train = cifar10_get['y_train']
y_test = cifar10_get['y_test']
input_shape = cifar10_get['input_shape']

# 학습된 모델 로드
# model = load_model('utils/save_models/cnn_mnist.h5')
model = load_model('utils/save_models/cnn_cifar10.h5')

# 모델 테스트
test_data_len = 3
test_data_index = np.random.choice(x_test.shape[0], test_data_len) # shuffle 및 데이터 선택
test_data = x_test[test_data_index]
predict_data = model.predict_classes(test_data)



if __name__ == '__main__':

    for i in range(test_data_len):
        (_x_train, _y_train), (_x_test, _y_test) = cifar10.load_data()
        plt.figure()
        plt.imshow(_x_test[test_data_index[i]])
        plt.colorbar()
        plt.grid(False)
        plt.show()

        print('Input : ' + str(np.argmax(y_test[test_data_index[i]]))
              + ', Predict : ' + str(predict_data[i]))
