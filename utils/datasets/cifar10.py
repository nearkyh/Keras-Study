from keras.datasets import cifar10
from keras.utils import np_utils


def load_data():
    return cifar10.load_data()

def pre_processing():
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = load_data()

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train /= 255
    x_test /= 255

    return {'num_classes':num_classes,
            'x_train':x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test,
            'input_shape': x_train.shape[1:]}



if __name__ == '__main__':

    print(pre_processing()['input_shape'])
    print(pre_processing()['y_test'])
