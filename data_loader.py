import keras
from keras.datasets import cifar10
import config

def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, config.num_classes)
    y_test = keras.utils.to_categorical(y_test, config.num_classes)

    return (x_train, y_train), (x_test, y_test)
