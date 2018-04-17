import os

import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.utils.vis_utils import plot_model

import data_loader
import config


(x_train, y_train), (x_test, y_test) = data_loader.load_data()

model = Sequential([
    Conv2D(32, (3,3),padding = 'same', activation = 'relu', input_shape= x_train.shape[1:]),
    Conv2D(32, (3,3 ), activation = 'relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    Conv2D(64, (3,3),padding = 'same', activation = 'relu'),
    Conv2D(64, (3,3 ), activation = 'relu'),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),
    Flatten(),
    Dense(512, activation = 'relu'),
    Dropout(0.5),
    Dense(config.num_classes, activation = 'softmax'),
])

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

print(model.summary())

rms = keras.optimizers.rmsprop(lr=0.0001, decay = 1e-6)

model.compile(loss ='categorical_crossentropy', optimizer = rms, metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

print('Fitting Model')
model.fit(x_train, y_train, batch_size = config.batch_size,
                            epochs = config.epochs,
                            validation_data = (x_test, y_test),
                            shuffle = True)

if not os.path.isdir(config.save_dir):
    os.makedirs(config.save_dir)
model_path = os.path.join(config.save_dir, config.model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
