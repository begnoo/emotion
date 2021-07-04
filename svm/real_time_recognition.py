import cv2
import _pickle as cPickle
import dlib
import argparse
import numpy as np
import feature_extraction
import cnn.predict as prd

from svm_params import TRAINING_PROPS, IMAGE_PROPS
from os import path

face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def load_model():
    print("loading model...")
    if path.isfile(TRAINING_PROPS.model_path):
        with open(TRAINING_PROPS.model_path, 'rb') as f:
            svm_model = cPickle.load(f)
            return svm_model
    else:
        print("Error: file '{}' not found".format(TRAINING_PROPS.model_path))
        exit(-1)


def get_features_from_image(img):
    rectangles = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
    landmarks = feature_extraction.get_landmarks(img, rectangles)
    hog_features = [feature_extraction.sliding_hog_windows(img)]
    fl_landmarks = landmarks.flatten()
    return np.concatenate((fl_landmarks, hog_features), axis=1)


def predict(svm_model, img):
    features = get_features_from_image(img)
    if not svm_model:
        print("Error: model not loaded")
    # noinspection PyBroadException
    try:
        return svm_model.predict(features)
    except:
        return ''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--method", default="cnn")  # cnn | svm
    args = parser.parse_args()

    if args.method != 'cnn' and args.method != 'svm':
        print("Invalid method selected")
        exit(-1)

    model = load_model()
    print("starting camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, current_image = cap.read()
        if not ret:
            continue
        gray_img = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(current_image, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest (face)
            roi_gray = cv2.resize(roi_gray, (48, 48))

            if args.method == 'svm':
                prediction = predict(model, roi_gray)
                emotion = prediction[0]
            else:
                if IMAGE_PROPS.DATASET == 'ck+':
                    emotion = prd.predict(roi_gray, 'ck')
                else:
                    emotion = prd.predict(roi_gray, 'fer')

            cv2.putText(current_image, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        resized_img = cv2.resize(current_image, (1000, 700))
        cv2.imshow('Window', resized_img)
        if cv2.waitKey(10) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
