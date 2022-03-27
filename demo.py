# demo.py
# File to perform classification on user-generated images
# ISL static fingerspelling letters. Proposed basis for recognition
# of users signing in real-time.
# 
# Used to demo project on 23rd March 2022
#
# Author: Ciara Sookarry
# Date: 20/03/22

import csv
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import os
import pandas as pd
import pickle
import re

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

counter = 0

# Apply landmarks to user images
# and display
user_imgs_loc = "/home/ciara/Documents/FYP/ISL_Translation/User_ISL"
user_imgs = os.listdir(user_imgs_loc)

for img in user_imgs:
    with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=1,
                model_complexity=0,
                min_detection_confidence=0.05) as hands:
        annotated = cv2.imread(f"{user_imgs_loc}/{img}")
        resized = cv2.resize(annotated, (800,600))
        results = hands.process(resized)

        if not results.multi_hand_landmarks:
            print(f"Landmarks could not be applied to {img}")
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            continue


        print(f"Applying landmarks to {img}")
        # Counter to control how many images we display
        counter += 1
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(21):
                mp_drawing.draw_landmarks(
                    resized,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        if (counter%13) == 0:
            #resized = cv2.resize(annotated, (800,600))
            cv2.imshow(f'Landmarked Image', resized)
            cv2.waitKey(0)


filename = 'real_time_svm_model.sav'
svclassifier = pickle.load(open(filename, 'rb'))

# Predict on real user data
real_test = pd.read_csv("real_time_test_landmarks.csv")
real_X_test = real_test.drop('label', axis=1)
real_y_test = real_test['label']

real_y_pred = svclassifier.predict(real_X_test)

print(classification_report(real_y_test, real_y_pred, zero_division=0))
ConfusionMatrixDisplay.from_predictions(real_y_test, real_y_pred, cmap='BuGn')

# Predict on testing set from dataset
test = pd.read_csv("testing_landmarks.csv")
X_test = test.drop('label', axis=1)
y_test = test['label']

y_pred = svclassifier.predict(X_test)
# print(y_pred)
# print(y_test)

print(classification_report(y_test, y_pred, zero_division=0))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='BuGn')

plt.show()
