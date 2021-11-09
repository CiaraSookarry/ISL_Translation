# split_sets.py
#
# Code to split each letter into training and testing sets
# then combine all training sets together and all testing 
# sets together to create overall training and testing sets
# 
# Author: Ciara Sookarry
# Date: 7 Nov 2021

import glob
import numpy

from PIL import Image
from sklearn.model_selection import train_test_split  # Splits data

# Import images as lists
a_list = glob.glob("/home/ciara/Documents/ISLDataset/ISL_50k/Frames_A/Person6*.jpg")
b_list = glob.glob("/home/ciara/Documents/ISLDataset/ISL_50k/Frames_B/Person6*.jpg")

# Make images fit X and Y
a_X = numpy.zeros((1, 402)) # There are 402 images of Person6 performing letter A
a_y = list()

#b_X = numpy.zeros((1, 307200))
#b_y = list("B")

for img in a_list:
    #img_pixels = list(Image.open(img).getdata())
    #print(len(img_pixels))
    #print(img)
    a_X = a_list
    a_y.append("A")

#for image in b_list:
#    image_pixels = list(Image.open(image).getdata())
    #print(len(img_pixels))
#    b_X = numpy.vstack((b_X, image_pixels))
#    b_y.append("B")

#print(f"a_X shape: {a_X.shape}, b_X shape: {b_X.shape}")
print(f"a_X shape: {len(a_X)}, a_y shape: {len(a_y)}")

# Split each list into training/testing
a_X_train,a_X_test, a_y_train, a_y_test = train_test_split(a_X, a_y)

print(f"a_X_train: {len(a_X_train)}")
print(f"a_X_test: {len(a_X_test)}")
print(f"a_y_train: {len(a_y_train)}")
print(f"a_y_test: {len(a_y_test)}")

#b_X_train,b_X_test, b_y_train, b_y_test = train_test_split(b_X, b_y)

#print(f"b_X_train: {len(b_X_train)}")
#print(f"b_X_test: {len(b_X_test)}")
#print(f"b_y_train: {len(b_y_train)}")
#print(f"b_y_test: {len(b_y_test)}")

# Combine lists together

