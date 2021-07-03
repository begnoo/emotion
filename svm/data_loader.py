import numpy as np
from svm.svm_params import *
from os import path


def load_data(validation_data=True, test_data=False):
    train_folder = path.join(IMAGE_PROPS.OUTPUT_FOLDER, 'training')
    validation_folder = path.join(IMAGE_PROPS.OUTPUT_FOLDER, 'validation')
    test_folder = path.join(IMAGE_PROPS.OUTPUT_FOLDER, 'test')

    data = {}
    validation = {}
    test = {}

    data['X'] = np.load(path.join(train_folder, 'landmarks.npy'))
    data['X'] = np.array([x.flatten() for x in data['X']])
    data['X'] = np.concatenate((data['X'], np.load(train_folder + '/hog_features.npy')), axis=1)
    data['Y'] = np.load(path.join(train_folder, 'labels.npy'))

    if validation_data:
        validation['X'] = np.load(path.join(validation_folder, 'landmarks.npy'))
        validation['X'] = np.array([x.flatten() for x in validation['X']])
        validation['X'] = np.concatenate((validation['X'], np.load(validation_folder + '/hog_features.npy')), axis=1)
        validation['Y'] = np.load(path.join(validation_folder, 'labels.npy'))

    if test_data:
        test['X'] = np.load(path.join(test_folder, 'landmarks.npy'))
        test['X'] = np.array([x.flatten() for x in test['X']])
        test['X'] = np.concatenate((test['X'], np.load(test_folder + '/hog_features.npy')), axis=1)
        test['Y'] = np.load(path.join(test_folder, 'labels.npy'))

    if not validation and not test:
        return data
    elif not test:
        return data, validation
    else:
        return data, validation, test
