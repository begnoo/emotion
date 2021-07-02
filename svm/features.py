import errno
import dlib
import numpy as np
import pandas as pd
import imageio
import cv2

from os import path, makedirs, listdir
from time import time
from skimage.feature import hog
import matplotlib.pyplot as plt

image_height = 48
image_width = 48
window_size = 24
window_step = 6
SELECTED_EMOTIONS = []
IMAGES_PER_EMOTION = 500  # -1 if there is no upper limit
LANDMARKS = True
HOG_FEATURES = True
HOG_WINDOWS_FEATURES = True
DATASET = 'fer2013'  # 'fer2013' | 'ck+' | 'ck+extracted'
OUTPUT_FOLDER = path.join(DATASET + '_features' + str(IMAGES_PER_EMOTION if IMAGES_PER_EMOTION >= 0 else ''))
IMAGES_PER_EMOTION = IMAGES_PER_EMOTION if IMAGES_PER_EMOTION >= 0 else float('inf')

# load dlib predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# create output folder
try:
    makedirs(OUTPUT_FOLDER)
except OSError as e:
    if e.errno == errno.EEXIST and path.isdir(OUTPUT_FOLDER):
        pass
    else:
        raise


def get_landmarks(image, rects):
    if len(rects) > 1:
        raise BaseException("TooManyFaces")
    if len(rects) == 0:
        raise BaseException("NoFaces")
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])


def get_new_label(label):
    return SELECTED_EMOTIONS.index(label)


def sliding_hog_windows(image):
    hog_windows = []
    for y in range(0, image_height, window_step):
        for x in range(0, image_width, window_step):
            window = image[y:y + window_size, x:x + window_size]
            hog_windows.extend(hog(window, orientations=8, pixels_per_cell=(8, 8),
                                   cells_per_block=(1, 1), visualize=False))
    return hog_windows


def get_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (image_width, image_height))


print("loading data...")
data = pd.DataFrame(columns=['emotion', 'image', 'usage'])
for category in ['training', 'test']:
    dataset_path = path.join('../' + DATASET, category)
    for emotion_dir in listdir(dataset_path):
        images_path = path.join(dataset_path, emotion_dir)
        training_counter = 1
        training_num = int(len(listdir(images_path)) * 0.75)
        for image_name in listdir(path.join(dataset_path, emotion_dir)):
            image_path = path.join(images_path, image_name)
            image = get_image(image_path)
            usage = 'test' if category == 'test' else 'training' if training_counter < training_num else 'validation'
            data = data.append({'emotion': emotion_dir, 'image': np.asarray(image), 'usage': usage}, ignore_index=True)

num_of_images_per_emotion = {}
SELECTED_EMOTIONS = data['emotion'].unique()

for category in data['usage'].unique():
    print('extracting features from ' + category + '...')
    start_time = time()
    # create folder
    if not path.exists(category):
        try:
            makedirs(path.join(OUTPUT_FOLDER, category))
        except OSError as e:
            if e.errno == errno.EEXIST and path.isdir(OUTPUT_FOLDER):
                pass
            else:
                raise

    # get samples and labels of the actual category
    category_data = data[data['usage'] == category]
    samples = category_data['image'].values
    labels = category_data['emotion'].values

    # get images and extract features
    images = []
    labels_list = []
    landmarks = []
    hog_features = []
    hog_images = []
    for i in range(len(samples)):
        try:
            emotion = labels[i]
            num_of_emotions = num_of_images_per_emotion.get(emotion, 0)
            if emotion in SELECTED_EMOTIONS and num_of_emotions < IMAGES_PER_EMOTION:
                image = samples[i].reshape((image_height, image_width))
                images.append(image)
                if HOG_WINDOWS_FEATURES:
                    features = sliding_hog_windows(image)
                    f, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                       cells_per_block=(1, 1), visualize=True)
                    hog_features.append(features)
                    hog_images.append(hog_image)
                elif HOG_FEATURES:
                    features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                              cells_per_block=(1, 1), visualize=True)
                    hog_features.append(features)
                    hog_images.append(hog_image)
                if LANDMARKS:
                    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
                    face_landmarks = get_landmarks(image, face_rects)
                    landmarks.append(face_landmarks)
                labels_list.append(emotion)
                num_of_images_per_emotion[emotion] = num_of_images_per_emotion.get(emotion, 0) + 1
        except Exception as e:
            print("error in image: " + str(i) + " - " + str(e))
    print('elapsed time: {}s'.format(time() - start_time))

    np.save(path.join(OUTPUT_FOLDER, category, 'images.npy'), images)
    np.save(path.join(OUTPUT_FOLDER, category, 'labels.npy'), labels_list)
    if LANDMARKS:
        np.save(path.join(OUTPUT_FOLDER, category, 'landmarks.npy'), landmarks)
    if HOG_FEATURES or HOG_WINDOWS_FEATURES:
        np.save(path.join(OUTPUT_FOLDER, category, 'hog_features.npy'), hog_features)
        np.save(path.join(OUTPUT_FOLDER, category, 'hog_images.npy'), hog_images)