import numpy as np
from pandas.io.parsers import read_csv
from matplotlib import pyplot as plt

learning_rate = 0.01
M = 1500

def load_csv(file_name):
    """
    Load the csv file. Returns numpy array
    """

    values = read_csv(file_name, header=None).values

    #always float
    return values.astype(float)

def h(x, _theta):
    return (np.dot(x, np.transpose(_theta))) #scalar product

def J(X, Y, m, _theta):
    return 1/(2*m) * np.sum((h(X, _theta) - Y)**2)

def minimize(X, Y, m, _theta):
    """
    Minimize using the given formulas
    """
    H = h(X, _theta)
    temp0 = _theta[0][0] - learning_rate*(1/m) * np.sum(H - Y)
    temp1 = _theta[0][1] - learning_rate*(1/m) * np.sum((H - Y) *  X)

    _theta[0, 0] = temp0
    _theta[0, 1] = temp1


def pinta_todo(X, Y, _theta):
    #plt.scatter(X, Y, 1, "red")
    plt.plot(X[:, 1:], h(X, _theta), color="grey")
    plt.show()

def gradient_descent_loop(X, Y, m):
    theta = np.array([[0, 0]], dtype=float)
    cost = np.array([], dtype=float)
    for i in range(M):
        minimize(X, Y, m, theta)
        np.append(cost, J(X, Y, m, theta))
    
    return theta, cost

plt.figure()
data = load_csv("ex1data1.csv")
X = data[:, :-1] #every col but the last
m = np.shape(X)[0] #number of training examples
Y = data[:, -1] #the last col, every row
Y = np.reshape(Y, (m, 1)) #dont know why this is needed, but it is
plt.scatter(X, Y, 1, "red")
X = np.hstack([np.ones([m, 1]), X]) 

theta, cost = gradient_descent_loop(X, Y, m)

pinta_todo(X, Y, theta)