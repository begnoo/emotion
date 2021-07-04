import os
import cv2
from cv2.cv2 import CascadeClassifier, imread

CK_TRAINING_PATH = "../ck+complete/training"
CK_TEST_PATH = "../ck+complete/test"

CK_EXTRACTED_TRAINING_PATH = "../ck+extracted/training"
CK_EXTRACTED_TEST_PATH = "../ck+extracted/test"

classifier = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

def extract_faces(folder_path, extracted_folder_path):
    for emotion in emotions:
        path = folder_path + "/" + emotion
        directory = os.fsencode(path)
        print(os.listdir(directory))
        for image_file in os.listdir(directory):
            image = imread(path + "/" + image_file.decode('utf-8'))
            bounding_boxes = classifier.detectMultiScale(image, scaleFactor=1.3, minNeighbors=3)

            if len(bounding_boxes) == 0:
                print("Can't find face, skipping...")
                print("check: " + image_file.decode('utf-8'))
                continue

            x, y, w, h = bounding_boxes[0]
            cropped = image[y:y + h, x:x + w]
            resized = cv2.resize(cropped, (48, 48), interpolation=cv2.INTER_AREA)
            new_image_path = extracted_folder_path + "/" + emotion + "/" + image_file.decode('utf-8')
            status = cv2.imwrite(new_image_path, resized)

            if len(bounding_boxes) > 1:
                print("check: " + new_image_path)

            if not status:
                print("Can't save " + new_image_path)
            else:
                print("[INFO] Object found. Saving locally.")


extract_faces(CK_TRAINING_PATH, CK_EXTRACTED_TRAINING_PATH)
extract_faces(CK_TEST_PATH, CK_EXTRACTED_TEST_PATH)