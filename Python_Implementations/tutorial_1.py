import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#Read training data into panda dataframe
training_data = pd.read_csv('training_data.csv', sep=',',header=None)
#Extract data from dataframe into numpy matrix
training_data = training_data.values

#Plot points read from csv file on to graph.
x = np.array([training_data[:,0]])
y = np.array([training_data[:,1]])
plt.plot(training_data[:,0], training_data[:,1], 'rx')

#transpose data read from file from a 1 dimentional arrays into vectors.
x_vector = x.transpose()
y_vector = y.transpose()

#Add column of ones to feature set.
all_ones = np.ones((x_vector.shape[0], 1))
features_set = np.append(all_ones, x_vector, axis=1)

#Create theta as a vector of zeros that has a size of the feature set. 
theta = np.zeros((features_set.shape[1], 1), dtype=int)

MAX_ITERATIONS = 1500
alpha = 0.07

#Adjust theta for 1500 iterations. 
for iterations in range(0, MAX_ITERATIONS):
    grad = np.dot((1/features_set.shape[0]), np.matmul(features_set.transpose(), np.subtract(np.matmul(features_set, theta), y_vector)))
    theta = np.subtract(theta, np.dot(alpha, grad))

#Get training data values read from csv to plot.
x_plot = training_data[:,0]
#Get y values to plot using the adjusted values of theta.
y_plot_data = (np.matmul(features_set, theta))
y_plot = y_plot_data[:,0]
plt.plot(x_plot, y_plot)
plt.show()







