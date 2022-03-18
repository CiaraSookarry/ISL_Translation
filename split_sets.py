# split_sets.py
#
# Code to split set of over 48,000 images into training and testing sets
# 
# Author: Ciara Sookarry
# Date: 7 Nov 2021

import fnmatch
import glob
import numpy
import random

from PIL import Image
from sklearn.model_selection import train_test_split    # Splits data
from termcolor import colored                           # Prints coloured text to terminal

#####################################
# Create input list and label list
#####################################
X_list = list()
y_list = list()

##########################
# Import images as lists
##########################
img_list = glob.glob("/home/ciara/Documents/ISLDataset/**/Frames_*/Person*", recursive=True) # will return 48345 image locations

#########################
# Create X and y lists
#########################
letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

for img in img_list:
    X_list = img_list
    # Cycle through letters and label images depending on file name
    for letter in letter_list:
        path = f"/home/ciara/Documents/ISLDataset/ISL_50k/Frames_{letter}"
        if fnmatch.fnmatch(img, f"{path}/*"):
            y_list.append(f"{letter}") 

######################################
# Split lists into training/testing
######################################
X_train,X_test, y_train, y_test = train_test_split(X_list, y_list, train_size=0.8, random_state=42)

##################
# Shuffle lists
##################
def getShuffleVar():
    return 0.3

random.shuffle(X_train, getShuffleVar)
random.shuffle(X_test, getShuffleVar)
random.shuffle(y_train, getShuffleVar)
random.shuffle(y_test, getShuffleVar)

################################################
# Check that lists have been shuffled properly
#################################################
print(coloured("Samples of shuffled X and y test sets", 'cyan'))
print(f"{y_test[:10]}\n")
print(f"{X_test[:10]}\n")
