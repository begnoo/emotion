from keras import Sequential, models
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.input_layer import InputLayer
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, ReLU, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy
from keras.regularizers import l2

FER_TRAINING_PATH = "../fer2013/train"
CK_TRAINING_PATH = "../ck+extracted/training"


def load_data(dataset):
    path = CK_TRAINING_PATH if dataset == "ck" else FER_TRAINING_PATH
    data_generator = ImageDataGenerator(validation_split=0.2)
    training_data = data_generator.flow_from_directory(directory=path,
                                                       target_size=(48, 48),
                                                       shuffle=True,
                                                       subset="training",
                                                       color_mode='grayscale')
    validation_data = data_generator.flow_from_directory(directory=path,
                                                         target_size=(48, 48),
                                                         shuffle=True,
                                                         subset="validation", color_mode='grayscale')
    return training_data, validation_data


def train_model(data):
    training_data, validation_data = data
    model = Sequential()
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1), 
                            kernel_regularizer=l2(0.01)))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(filters=128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(filters=256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=7, activation='softmax'))
    
    checkpoint = ModelCheckpoint("cnn_1.h5", monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x=training_data,
                        validation_data=validation_data,
                        epochs=100,
                        batch_size=32,
                        callbacks=[checkpoint, early])
    model.save("/model_with_" + dataset + ".h5")
    show_history(history)

    return model


def show_history(history):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    data = load_data('ck')
    model = train_model(data)
