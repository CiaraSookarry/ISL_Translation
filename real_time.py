# real_time.py
# File to perform classification on user-generated images
# ISL static fingerspelling letters. Proposed basis for recognition
# of users signing in real-time.
#
# Author: Ciara Sookarry
# Date: 20/01/22

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

landmarks = list()
label_list = list()

# Write landmark values to CSV file
def write_csv(data):
    header = ['WristX', 'WristY', 'ThumbCMCX', 'ThumbCMCY', 'ThumbMCPX', 'ThumbMCPY', 'ThumbIPX', 'ThumbIPY', 'ThumbTIPX', 'ThumbTIPY', 'IndexMCPX', 'IndexMCPY', 'IndexPIPX', 'IndexPIPY', 'IndexDIPX', 'IndexDIPY', 'IndexTIPX', 'IndexTIPY', 'MiddleMCPX', 'MiddleMCPY', 'MiddlePIPX', 'MiddlePIPY', 'MiddleDIPX', 'MiddleDIPY', 'MiddleTIPX', 'MiddleTIPY', 'RingMCPX', 'RingMCPY', 'RingPIPX', 'RingPIPY', 'RingDIPX', 'RingDIPY', 'RingTIPX', 'RingTIPY', 'PinkyMCPX', 'PinkyMCPY', 'PinkyPIPX', 'PinkyPIPY', 'PinkyDIPX', 'PinkyDIPY', 'PinkyTIPX', 'PinkyTIPY', 'label']

    with open('real_time_test_landmarks.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header) # Write in header titles
        writer.writerows(data)  # Write in landmark values for multiple images

###############
# Main code
###############
# List all user generated images
# user_imgs_loc = "/home/ciara/Documents/FYP/ISL_Translation/My_ISL"
user_imgs_loc = "/home/ciara/Documents/FYP/ISL_Translation/My_ISL/User_ISL"
user_imgs = os.listdir(user_imgs_loc)

for img in user_imgs:

    image = cv2.imread(f"{user_imgs_loc}/{img}")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, thresh4 = cv2.threshold(grey, thresh_val, 255, cv2.THRESH_TOZERO_INV)
    # annotated = cv2.cvtColor(thresh4, cv2.COLOR_GRAY2RGB)
    annotated = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB) # Convert image back to RGB because MediaPipe requires RGB image

    # cv2.imshow('Original Image', image)
    # cv2.imshow('Greyscale Image', grey)
    # cv2.imshow('Thresholded Image', thresh4)
    # cv2.imshow('Thresholded to RGB Image', annotated)
    # cv2.imshow('Grey to RGB Image', annotated)

    # Try to apply landmarks to image
    with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.05) as hands:
        results = hands.process(annotated)

        sign_lmarks = list()
        if not results.multi_hand_landmarks:
            print(f"Landmarks could not be applied to {img}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            continue

        
        print(f"Applying landmarks to {img}")
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(21):
                # Append x and y finger tip landmarks to landmark file for a single sign
                sign_lmarks.append(hand_landmarks.landmark[i].x)
                sign_lmarks.append(hand_landmarks.landmark[i].y)

            mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            
            # cv2.imshow('Landmarked Image', annotated)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
            # Extract image label from image name
            # Then append label to landmark file for single file
            # label = re.search('(.+?)_(.+?)_[0-9]', img)
            label = re.search('(.+?)_[0-9]', img)
            if label:
                label_letter = label.group(1)
                print(label_letter)
                sign_lmarks.append(label_letter)
            
            # Append landmarks for single sign to
            # list holding landmarks for all signs
            landmarks.append(sign_lmarks)

# Write all landmark to CSV
write_csv(landmarks)

# Attempt classification of user images
filename = 'svm_model.sav'
svclassifier = pickle.load(open(filename, 'rb'))

test = pd.read_csv("real_time_test_landmarks.csv")
X_test = test.drop('label', axis=1)
y_test = test['label']

y_pred = svclassifier.predict(X_test)

# print(y_pred)
# print(y_test)

print(classification_report(y_test, y_pred, zero_division=0))
print("\n")
letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']
# print(confusion_matrix(y_test, y_pred, labels=letter_list))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='BuGn')
plt.show()
