# We'll start by importing a few libraries that will make it easy to work with most machine learning projects.

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

# For a simple linear example, we'll just make some dummy data and that will act in the place of importing a dataset.

# linear data
X = np.array([1, 5, 1.5, 8, 1, 9, 7, 8.7, 2.3, 5.5, 7.7, 6.1])
y = np.array([2, 8, 1.8, 8, 0.6, 11, 10, 9.4, 4, 3, 8.8, 7.5])

#The reason we're working with numpy arrays is to make the matrix operations faster because they use less memory than Python lists. You could also take advantage of typing the contents of the arrays. Now let's take a look at what the data look like in a plot:

# show unclassified data
plt.scatter(X, y)
plt.show()

#Once you see what the data look like, you can take a better guess at which algorithm will work best for you. Keep in mind that this is a really simple dataset, so most of the time you'll need to do some work on your data to get it to a usable state.

#We'll do a bit of pre-processing on the already structured code. This will put the raw data into a format that we can use to train the SVM model.

# shaping data for training the model
training_X = np.vstack((X, y)).T
training_y = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]
#Now we can create the SVM model using a linear kernel.

# define the model
clf = svm.SVC(kernel='linear', C=1.0)
#That one line of code just created an entire machine learning model. Now we just have to train it with the data we pre-processed.

# train the model
clf.fit(training_X, training_y)
#That's how you can build a model for any machine learning project. The dataset we have might be small, but if you encounter a real-world dataset that can be classified with a linear boundary this model still works.

#With your model trained, you can make predictions on how a new data point will be classified and you can make a plot of the decision boundary. Let's plot the decision boundary.

# get the weight values for the linear equation from the trained SVM model
w = clf.coef_[0]

# get the y-offset for the linear equation
a = -w[0] / w[1]

# make the x-axis space for the data points
XX = np.linspace(0, 13)

# get the y-values to plot the decision boundary
yy = a * XX - clf.intercept_[0] / w[1]

# plot the decision boundary
plt.plot(XX, yy, 'k-')

# show the plot visually
plt.scatter(training_X[:, 0], training_X[:, 1], c=training_y)
plt.legend()
plt.show()
