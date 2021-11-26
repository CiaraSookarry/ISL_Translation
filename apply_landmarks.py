# apply_landmarks.py
#
# Program to apply mediapipe hands landmarks to dataset 
# images. Training and testing sets imported from 
# split_sets.py. Running this file also runs split_sets.py.
#
# Author: Ciara Sookarry
# Date: 20th November 2021

import cv2
import math
import mediapipe as mp
import numpy as np

from split_sets import X_train, X_test, y_train, y_test

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#############################
# Declare functions
#############################

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize_and_show(image, show):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

    if show == 1:
        cv2.imshow('Image' ,img)
        cv2.waitKey(1)
    #  cv2.destroyAllWindows()

# Split large sets of images into smaller batches for processing
# Ensures we don't run out of memory
def create_batches(list_name, batch_size):
    for i in range(0, len(list_name), batch_size):
        yield list_name[i:i + batch_size]

############################
# Main code
############################

# Read images with OpenCV.
batches = create_batches(X_test, 2000)
for sets in batches:
    images = {name: cv2.imread(name) for name in sets}
    print("Images read")
    # Preview the images.
    # for name, image in images.items():
        # print(name)       
        # resize_and_show(image, 0)
    # print("Images resized")
    # Run MediaPipe Hands.
    no_marks = 0
    marks = 0
    with mp_hands.Hands(
                    static_image_mode=True,
                    max_num_hands=1,
                    model_complexity=0,
                    min_detection_confidence=0.05) as hands:
        for name, image in images.items():
                    # Convert the BGR image to RGB, flip the image around y-axis for correct 
                    # handedness output and process it with MediaPipe Hands.
                    results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
                    '''
                    # Print handedness (left v.s. right hand).
                    print(f'Handedness of {name}:')
                    print(results.multi_handedness)
                    '''
                    if not results.multi_hand_landmarks:
                        no_marks += 1
                        continue
                    # Draw hand landmarks of each hand.
                    marks += 1
                    print(f'Hand landmarks of {name}:')
                    image_hight, image_width, _ = image.shape
                    annotated_image = cv2.flip(image.copy(), 1)
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Print index finger tip coordinates.
                        '''  
                        print(
                                        f'Index finger tip coordinate: (',
                                        f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
                                        f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight})'
                        )
                        
                        
                        mp_drawing.draw_landmarks(
                                        annotated_image,
                                        hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style())
                    resize_and_show(cv2.flip(annotated_image, 1), 1)
                    '''
    print("Landmarks Applied")
    # Print number of images to which landmarks 
    # couldn't be applied
    print(no_marks)
    print(marks)

    '''
    DESIRED_HEIGHT = 480
    DESIRED_WIDTH = 480
    def resize_and_show(image, show):
        h, w = image.shape[:2]
        if h < w:
            img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
        else:
            img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))

        if show == 1:
            cv2.imshow('Image' ,img)
            cv2.waitKey(1)
        #  cv2.destroyAllWindows()

    # Split large sets of images into smaller batches for processing
    # Ensures we don't run out of memory
    def create_batches(list_name, batch_size):
        for i in range(0, len(list_name), batch_size):
            yield list_name[i:i + batch_size]
    '''
