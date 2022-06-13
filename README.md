# ISL_Translation
Final year project to facilitate translation of Irish Sign Language (ISL) fingerspelling alphabet to English. Uses data set found here: https://github.com/marlondcu/ISL_50k. Landmarks applied to images of hands using MediaPipe Hands model: https://google.github.io/mediapipe/solutions/hands.html.

## Operation Outline
### Organising Dataset
The images from https://github.com/marlondcu/ISL_50k are split into 6 folders depending on which Person{1...6} performed the sign. This is not useful for our application. So a script was written to sort the images into different folders depending on which sign the image contains as opposed to who signed it. This script contained the code below:
```
letter = 'A' # This must be changed manually 
os.mkdir(f"/home/ciara/Documents/ISLDataset/ISL_50k/Frames_{letter}") # Create new directory for each letter

for n in range(1,7):
    person = f"Person{n}"

    source = f"/home/ciara/Documents/ISLDataset/ISL_50k/{person}"
    destination = f"/home/ciara/Documents/ISLDataset/ISL_50k/Frames_{letter}"

    files = os.listdir(source)

    for file in files:
        if fnmatch.fnmatch(file, f"{person}-{letter}-*"):
            new_path = shutil.copy(f"{source}/{file}",f"{destination}/{file}")
            print(new_path)
```

The code below can be found in `split_sets.py` and it creates `X_list` (list containing all image locations) and `y_list` (list with corresponding letter labels).

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
`apply_landmarks.py` is used to apply the MediaPipe hand landmarks to the dataset images in batches and write the landmark values to CSV files called training.csv and testing.csv.

### Classification Model
A Support Vector Machine (SVM) was used to classify the signs using the landmarks and labels in the CSV files. Other machine learning models were not considered because the SVM was able to provide >90% precision and recall overall (even with a linear kernel!) which was deemed adequate. 

## Optimizing Hyperparameters
Using the correct hyperparameters is vital to ensure that the full poterntial of a machine learning model is reached. These hyperparamters can be altered manually and assessed but this is not efficient. Better ways to optimise hyperparmeters can be seen below.

### Random Search
Hyperparameters are randomly chosen from specified distributions using `RandomizedSearchCV` from scikitlearn. This is done `n_iter` times. The distributions used in this project can be seen below:
```
param_distributions = dict(
        kernel    = ['linear', 'rbf', 'sigmoid'],
        C         = uniform(loc=1, scale=99),
        gamma     = uniform(loc=1e-3, scale=1e3),
        coef0     = uniform(loc=-10, scale=20)
    )
```

### Exhaustive Grid Search
With this method candidate hyperparameters are specified in a grid and `GridSearchCV` from scikitlearn can be used to try each possible combination of hyperparameters using cross validation and return the best hyperparameters. The code below shows the parameters that were used in this project.
```
35     tuned_parameters = [
36     {"kernel": ["linear"], "C": [1, 10, 100, 1000, 10000]},
37     {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [10, 100, 1000]},
38     {"kernel": ["sigmoid"], "gamma": ["auto", "scale", 1e-3], "C": [10, 100, 1000]},
39     ]
```
### Bayesian Hyperparamter Optimization
Bayesian optimization methods are much more sophisticated than either grid or random search because they are **informed** unlike the others. This means that Bayesian methods hold onto a history of past evaluation scores and the hyperparameters that correspond to each score. The history is then used to create a probabilistic surrogate model of the cost function, p(y|x),  that is cheaper to evaluate than the objective cost function. Hyperparameters are tried out on the surrogate and only the best performing hyperparameters are evaluated on the objective function. Once the hyperparameters have been evaluated on the objective function, the results are added to the history and used to update the surrogate model, making it more accurate. 

This project used the `tpe` surrogate model from `hyperopt` as can be seen in the file `bayes_svm.py`. These were the hyperparameter distributions used:
```
space4svm = {
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'rbf']),
    'C': hp.uniform('C', 1, 100),
    'gamma': hp.uniform('gamma', 1e-3, 1e3),
    'coef0': hp.uniform('coef0', -10, 10)
}
```

## Demo
The file `demo.py` was used in the final presentation of this project. It applies landmarks to images not found in the dataset at https://github.com/marlondcu/ISL_50k  and attempts to classify the signs found within. The results are displayed in a confusion matrix and compared with the confusion matrix generated using images from the original dataset.

## Future Work
1. Real-Time Recognition
This was attempted but not completed due to the time-constraints of the project. The ability to recognise a users signs in real-time would take the project to the next level.

2. Recognition of dynamic signs
The 3 signs of the fingerspelling alphabet that require movement were left out of this project. Their recognition would require the use of videos from https://github.com/marlondcu/ISL_50k.
