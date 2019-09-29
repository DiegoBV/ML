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
    return (_theta[0] + x*_theta[1])

def J(_cases, _theta):
    m = len(_cases)
    return 1/(2*m) * np.sum((h(_cases[:, 0], _theta) - _cases[:, 1])**2)

def minimize(_cases, _theta):
    """
    Minimize using the given formulas
    """
    m = len(_cases)
    temp0 = _theta[0] - learning_rate*(1/m) * np.sum(h(_cases[:, 0], _theta) - _cases[:,1])
    temp1 = _theta[1] - learning_rate*(1/m) * np.sum((h(_cases[:, 0], _theta) - _cases[:,1]) * _cases[:, 0])

    _theta[0] = temp0
    _theta[1] = temp1


def pinta_todo(_cases, _theta):
    plt.figure()
    plt.scatter(_cases[:,0], _cases[:,1], 1, "red")
    plt.plot(_cases[:,0], h(_cases[:,0], _theta), color="grey")
    plt.show()

def gradient_descent_loop():
    theta = [0,0]
    cases = load_csv("ex1data1.csv")

    for i in range(M):
        minimize(cases, theta)
        print(J(cases, theta))
    
    pinta_todo(cases, theta)

gradient_descent_loop()