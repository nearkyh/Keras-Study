from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K


def load_data():
    return mnist.load_data()

def pre_processing():
    # 데이터셋 레이블 갯수
    num_classes = 10

    # 입력 이미지 크기
    img_rows, img_cols = 28, 28

    # mnist 데이터셋 로드
    (x_train, y_train), (x_test, y_test) = load_data()

    # tensorflow 백엔드 경우 channels_last
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # 형병환
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    # 정규화
    x_train /= 255
    x_test /= 255

    # 원-핫 벡터
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    return {'num_classes':num_classes,
            'x_train':x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test,
            'input_shape': input_shape}



if __name__ == '__main__':

    print(pre_processing()['input_shape'])
