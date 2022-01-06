# ISL_Translation
Final year project to facilitate translation of Irish Sign Language (ISL) fingerspelling alphabet to English. Uses data set found here: https://github.com/marlondcu/ISL_50k. Landmarks applied to images of hands using MediaPipe Hands model: https://google.github.io/mediapipe/solutions/hands.html.

## Operation Outline
### Organising Dataset
The images from https://github.com/marlondcu/ISL_50k are split into 6 folders depending on which Person{1...6} performed the sign. This is not useful for our application. So a script was written to sort the images into different folders depending on which sign the image contains as opposed to who signed it. The code below can be found in `split_sets.py`

```
33 letter_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']
34 
35 for img in img_list:
36     X_list = img_list
37     # Cycle through letters and label images depending on file name
38     for letter in letter_list:
39         path = f"/home/ciara/Documents/ISLDataset/ISL_50k/Frames_{letter}"
40         if fnmatch.fnmatch(img, f"{path}/*"):
41             y_list.append(f"{letter}")
```
### Creating Training and Testing Sets
Before any classification can take place, we must create the sets for training the model and testing its accuracy. The `train_test_split()` method from scikitlearn is used to create the x and y lists for both training and testing given the proportion of images to be sent to the training set. At this stage, the sets are made of file locations for each image.
```
X_train,X_test, y_train, y_test = train_test_split(X_list, y_list, train_size=0.8, random_state=42)
```

The sets are then shuffled to disperse the different letters all throughout the sets.
```
64 def getShuffleVar():
65     return 0.3
66 
67 random.shuffle(X_train, getShuffleVar)
68 random.shuffle(X_test, getShuffleVar)
69 random.shuffle(y_train, getShuffleVar)
70 random.shuffle(y_test, getShuffleVar)
```

### Applying Landmarks

### Writing Landmarks to CSV format

### Optimizing Hyperparameters
#### Exhaustive Grid Search
#### Bayesian Hyperparamter Optimization
