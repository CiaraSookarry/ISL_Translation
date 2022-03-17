# img_process.py
# File to perform image processing necessary for real-time
# classification of ISL static fingerspelling letters
# 
# Author: Ciara Sookarry
# Date: 20/01/22

import csv
import cv2
import mediapipe as mp
import os
import pandas as pd
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

thresh_val = 160
landmarks = list()

# Write landmark values to CSV file
def write_csv(data):
    header = ['WristX', 'WristY', 'ThumbCMCX', 'ThumbCMCY', 'ThumbMCPX', 'ThumbMCPY', 'ThumbIPX', 'ThumbIPY', 'ThumbTIPX', 'ThumbTIPY', 'IndexMCPX', 'IndexMCPY', 'IndexPIPX', 'IndexPIPY', 'IndexDIPX', 'IndexDIPY', 'IndexTIPX', 'IndexTIPY', 'MiddleMCPX', 'MiddleMCPY', 'MiddlePIPX', 'MiddlePIPY', 'MiddleDIPX', 'MiddleDIPY', 'MiddleTIPX', 'MiddleTIPY', 'RingMCPX', 'RingMCPY', 'RingPIPX', 'RingPIPY', 'RingDIPX', 'RingDIPY', 'RingTIPX', 'RingTIPY', 'PinkyMCPX', 'PinkyMCPY', 'PinkyPIPX', 'PinkyPIPY', 'PinkyDIPX', 'PinkyDIPY', 'PinkyTIPX', 'PinkyTIPY', 'label']

    with open('real_time_test_landmarks.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        # print(f'Data: {data}\n')
        writer.writerows(data) # multiple letters in one file
        # writer.writerow(data)

user_imgs = os.listdir("/home/ciara/Documents/FYP/ISL_Translation/My_ISL")

for img in user_imgs:

    # image = cv2.imread('/home/ciara/Documents/FYP/ISL_Translation/My_ISL/A_Wall_3.jpg')
    image = cv2.imread(f"/home/ciara/Documents/FYP/ISL_Translation/My_ISL/{img}")
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ret, thresh4 = cv2.threshold(grey, thresh_val, 255, cv2.THRESH_TOZERO_INV)
    # annotated = cv2.cvtColor(thresh4, cv2.COLOR_GRAY2RGB)
    annotated = cv2.cvtColor(grey, cv2.COLOR_GRAY2RGB)

    # cv2.imshow('Original Image', image)
    # cv2.imshow('Greyscale Image', grey)
    # cv2.imshow('Thresholded Image', thresh4)
    # cv2.imshow('Thresholded to RGB Image', annotated)
    # cv2.imshow('Grey to RGB Image', annotated)

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


        #annotated_image = cv2.flip(image.copy(), 1)    
        
        print(f"Applying landmarks to {img}")
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(21):
                # Append x and y finger tip landmarks 
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
            sign_lmarks.append(427)
            landmarks.append(sign_lmarks)

write_csv(landmarks)

filename = 'svm_model.sav'
svclassifier = pickle.load(open(filename, 'rb'))

test = pd.read_csv("real_time_test_landmarks.csv")
X_test = test.drop('label', axis=1)
y_test = test['label']

y_pred = svclassifier.predict(X_test)
print(y_pred)


