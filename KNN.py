# Let’s start with importing libraries:
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt #data visualization

from sklearn.datasets import make_blobs #synthetic dataset
from sklearn.neighbors import KNeighborsClassifier #kNN classifier
from sklearn.model_selection import train_test_split #train and test sets

# Scikit-learn provides many useful functions to create synthetic datasets which are very helpful for practicing machine learning algorithms. I will use make_blobs function.
#create synthetic dataset
X, y = make_blobs(n_samples = 100, n_features = 2, centers = 4,
                       cluster_std = 1.5, random_state = 4)


#This code creates a dataset with 100 samples divided into 4 classes and the number of features is 2. Number of samples, features and classes can easily be adjusted using related parameters. We can also adjust how much each cluster (or class) is spread. Let’s visualize this synthetic data set:
#scatter plot of dataset
plt.figure(figsize = (10,6))
plt.scatter(X[:,0], X[:,1], c=y, marker= 'o', s=50)
plt.show()

# For any supervised machine learning algorithm, it is very important to divide dataset into train and test sets. We first train the model and test it using different parts of dataset. If this separation is not done, we basically test the model with some data it already knows. We can easily do this separation using train_test_split function.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
#We can specify how much of the original data is used for train or test sets using train_size or test_size parameters, respectively. Default separation is 75% for train set and 25% for test set.
#Then we create a kNN classifier object. To show the difference between the importance of k value, I create two classifiers with k values 1 and 5. Then these models are trained using train set. n_neighbors parameter is used to select k value. Default value is 5 so it does not have to be explicitly written.
knn5 = KNeighborsClassifier() #k=5
knn1 = KNeighborsClassifier(n_neighbors=1) #k=1

knn5.fit(X_train, y_train)
knn1.fit(X_train, y_train)

#Then we predict the target values in the test set and compare with actual values.
y_pred_5 = knn5.predict(X_test)
y_pred_1 = knn1.predict(X_test)

#In order to see the effect of k values, let’s visualize test set and predicted values with k=5 and k=1.
from sklearn.metrics import accuracy_score
print("Accuracy of kNN with k=5", accuracy_score(y_test, y_pred_5))
print("Accuracy of kNN with k=1", accuracy_score(y_test, y_pred_1))

#Original Data
plt.figure(figsize = (10,6))
plt.scatter(X_test[:,0], X_test[:,1], c=y_test, marker= 'o', s=50)
plt.title("Original data", fontsize=20)
plt.show()

#Predicted values with k=5
plt.figure(figsize = (10,6))
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred_5, marker= 'o', s=50)
plt.title("Predicted values with k=5", fontsize=20)
plt.show()

#Predicted values with k=1
plt.figure(figsize = (10,6))
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred_1, marker= 'o', s=50)
plt.title("Predicted values with k=1", fontsize=20)
plt.show()

### How to find the best k value????
# k=1: The model is too specific and not generalized well. It also tends to be sensitive to noise. The model accomplishes a high accuracy on train set but will be a poor predictor on new, previously unseen data points. Therefore, we are likely to end up with an overfit model.
# k=100: The model is too generalized and not a good predictor on both train and test sets. This situation is known as underfitting.
# How do we find the optimum k value? Scikit-learn provides GridSearchCV function that allows us to easily check multiple values for k. Let’s go over an example using a dataset available under scikit-learn datasets module.

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV

cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

#After we import the required libraries and load the dataset, we can create a GridSearchCV object.
knn_grid = GridSearchCV(estimator = KNeighborsClassifier(), 
                        param_grid={'n_neighbors': np.arange(1,20)}, cv=5)
                        
knn_grid.fit(X_cancer, y_cancer)

#We do not need to split the the datasets because cv parameter splits the dataset. The default value for cv parameter is 5 but I explicitly wrote it to emphasize why we don’t need to use train_test_split.
#cv=5 basically splits the dataset into 5 subsets. GridSearchCV does 5 iterations and use 4 subsets for training and 1 subset for testing at each time. In this way, we are able to use all data points for both training and testing.
#We can check which parameters give us best results using best_params_ method:
knn_grid.best_params_
