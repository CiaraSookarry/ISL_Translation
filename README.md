# ISL_Translation
Final year project to facilitate translation of Irish Sign Language (ISL) fingerspelling alphabet to English. Uses data set found here: https://github.com/marlondcu/ISL_50k. Landmarks applied to images of hands using MediaPipe Hands model: https://google.github.io/mediapipe/solutions/hands.html.

Dataset split into training (80%), validation (10%), and testing (10%) like so:

```
X = df.drop(columns = ['SalePrice']).copy()
y = df['SalePrice']

# In the first step we will split the data in training and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)

# Now since we want the valid and test size to be equal (10% each of overall data). 
# we have to define valid_size=0.5 (that is 50% of remaining data)
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)
```
