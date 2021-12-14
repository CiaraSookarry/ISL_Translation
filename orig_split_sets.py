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
import random

from PIL import Image
from sklearn.model_selection import train_test_split  # Splits data

##########################
# Import images as lists
##########################
a_list = glob.glob("/home/ciara/Documents/ISLDataset/ISL_50k/Frames_A/Person6*.jpg")
b_list = glob.glob("/home/ciara/Documents/ISLDataset/ISL_50k/Frames_B/Person6*.jpg")

#########################
# Create X and y lists
#########################
a__X = numpy.zeros((1, 402)) # There are 402 images of Person6 performing letter A
a_y = list()

b_X = numpy.zeros((1, 370))
b_y = list()

for img in a_list:
    a_X = a_list
    a_y.append("A")

for image in b_list:
    b_X = b_list
    b_y.append("B")

print(f"a_X shape: {len(a_X)}, a_y shape: {len(a_y)}, b_X shape: {len(b_X)}, b_y shape: {len(b_y)}\n")

##########################################
# Split each list into training/testing
##########################################
a_X_train,a_X_test, a_y_train, a_y_test = train_test_split(a_X, a_y, train_size=0.8, random_state=42)

print(f"a_X_train: {len(a_X_train)}")
print(f"a_X_test: {len(a_X_test)}")
print(f"a_y_train: {len(a_y_train)}")
print(f"a_y_test: {len(a_y_test)}\n")

b_X_train,b_X_test, b_y_train, b_y_test = train_test_split(b_X, b_y, train_size=0.8, random_state=42)

print(f"b_X_train: {len(b_X_train)}")
print(f"b_X_test: {len(b_X_test)}")
print(f"b_y_train: {len(b_y_train)}")
print(f"b_y_test: {len(b_y_test)}\n")

###########################
# Combine lists together
###########################
X_train = a_X_train + b_X_train
X_test = a_X_test + b_X_test
y_train = a_y_train + b_y_train
y_test = a_y_test + b_y_test

print(f"X_train: {len(X_train)}")
print(f"X_test: {len(X_test)}")
print(f"y_train: {len(y_train)}")
print(f"y_test: {len(y_test)}\n")
print(f"{y_test}\n")

##########################
# Shuffle lists
##########################
def getShuffleVar():
    return 0.3

random.shuffle(X_train, getShuffleVar)
random.shuffle(X_test, getShuffleVar)
random.shuffle(y_train, getShuffleVar)
random.shuffle(y_test, getShuffleVar)

print(f"{y_test}\n")
#print(f"{X_test}\n")
