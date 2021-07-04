import numpy as np
from keras import models
from keras_preprocessing.image import ImageDataGenerator

FER_TEST_PATH = "../fer2013/test"
CK_TEST_PATH = "../ck+/test"

fer_labels = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
ck_labels = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]


def load(dataset):
    return models.load_model('model_with_' + dataset + '.h5')


def predict(image, dataset):
    model = load(dataset)
    reshaped_image = np.array(image, dtype=np.float32)
    reshaped_image = np.reshape(reshaped_image, (-1, 48, 48, 1))
    prediction_classes = model.predict(reshaped_image)
    prediction = prediction_classes.argmax()
    return ck_labels[prediction] if dataset == 'ck' else fer_labels[prediction]


def evaluate(dataset):
    path = CK_TEST_PATH if dataset == "ck" else FER_TEST_PATH
    test_data = ImageDataGenerator().flow_from_directory(directory=path,
                                                         target_size=(48, 48),
                                                         shuffle=True, color_mode='grayscale')
    model = load(dataset)
    model.evaluate(test_data)


if __name__ == '__main__':
    evaluate("fer")
