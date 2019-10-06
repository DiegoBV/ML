import numpy as np
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys

learning_rate = 0.01

def load_csv(file_name):
    """
    Load the csv file. Returns numpy array
    """

    values = read_csv(file_name, header=None).values

    #always float
    return values.astype(float)

def h(x, _theta):
    return (np.dot(x, _theta)) #scalar product

def normal_equation(X, Y):
    inv = np.linalg.pinv((np.dot(np.transpose(X), X))) #(X^T * X)^-1
    transp = np.dot(np.transpose(X), Y) #X^T * Y
    return np.dot(inv, transp)

def adapt_user_values(user_values):
    user_values = np.hstack([[1], user_values])
    return user_values

data = load_csv("ex1data2.csv")
X = data[:, :-1] #every col but the last
m = np.shape(X)[0] #number of training examples
n = np.shape(X)[1]
Y = data[:, -1] #the last col, every row
Y = np.reshape(Y, (m, 1)) #dont know why this is needed, but it is
X = np.hstack([np.ones([m, 1]), X])

theta = normal_equation(X, Y)

print("Values of theta: " + str(theta))
user_values = np.array(list(map(float, input("Enter query values: ").split())), dtype=float)
user_values = adapt_user_values(user_values)
print("Your prediction: " + str(int(h(user_values, theta))))
