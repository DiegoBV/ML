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
    return (np.dot(x, np.transpose(_theta))) #scalar product

def J(X, Y, _theta):
    m = np.shape(X)[0]
    return 1/(2*m) * np.sum((h(X, _theta) - Y)**2)

def minimize(X, Y, m, n, _theta):
    """
    Minimize using the given formulas
    """
    H = h(X, _theta)
    for i in range(n + 1):
        columnX = np.reshape(X[:, i], (m, 1)) #X[:, i] devuleve una fila con los datos de la columna, hay que hacer reshape para que devuelva los datos en filas separadas 
        aux = np.sum((H - Y) * columnX)
        temp = _theta[0][i] - learning_rate*(1/m) * aux
        _theta[0, i] = temp

    """temp0 = _theta[0][0] - learning_rate*(1/m) * np.sum(H - Y)
    temp1 = _theta[0][1] - learning_rate*(1/m) * np.sum((H - Y) *  X)

    _theta[0, 0] = temp0
    _theta[0, 1] = temp1"""

def make_paint_data(X, Y):
    step= 0.1
    Theta0 = np.arange(-10, 10, step)
    Theta1 = np.arange(-1, 4, step)

    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)

    coste = np.empty_like(Theta0)

    for ix, iy in np.ndindex(Theta0.shape):
        coste[ix, iy] = J(X, Y, [Theta0[ix, iy], Theta1[ix, iy]])

    return Theta0, Theta1, coste

def normalize(X, n):
    """
    normalize the matrix given using the mean and deviation of every column (attribute)
    returns the normalized matrix, the mu vector (mean of every attribute) and sigma vector (deviation of every attribute)
    """
    X_norm = np.empty_like(X)
    sigma = np.empty(n + 1)
    mu = np.empty(n + 1)
    for i in range(n + 1):
        mu[i] = np.mean(X[:, i])
        sigma[i] = np.std(X[:, i])
        if sigma[i] != 0:
            new_value = (X[:, i] - mu[i])/sigma[i]
            X_norm[:, i] = new_value
        else:
            X_norm[:, i] = 1 #??
    #sigma & mu???"""
    return X_norm, mu, sigma

def normalize_user_values(user_values, mu, sigma):
    new_data = (user_values - mu[1:])/sigma[1:]
    new_data = np.hstack([[1], new_data])
    return new_data

def draw_points_plot(X, Y, _theta):
    plt.figure()
    plt.scatter(X[:, 1], Y, 1, "red")
    plt.plot(X[:, 1:], h(X, _theta), color="grey")
    plt.show()

def draw_cost_3D(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d') 

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)  

    fig.colorbar(surf, shrink =0.5, aspect=5)
    plt.show()

def draw_contour(X, Y, Z):
    plt.figure()
    plt.contour(X, Y, Z)
    plt.show()

def draw_cost(cost):
    plt.figure()
    X = np.linspace(0, 400, len(cost))
    plt.plot(X, cost)
    plt.show()
    
def gradient_descent_loop(X, Y, m, n):
    theta = np.zeros([1, n + 1], dtype=float)
    cost = np.array([], dtype=float)
    auxCost = sys.maxsize
    while True:
        minimize(X, Y, m, n, theta)
        cost = np.append(cost, J(X, Y, theta))
        if abs(auxCost - cost[-1]) < 1e-4:
            #Stops the loop when we reach the convergence value of 10^-4
            break
        auxCost = cost[-1]
    
    return theta, cost

data = load_csv("ex1data2.csv")
X = data[:, :-1] #every col but the last
m = np.shape(X)[0] #number of training examples
n = np.shape(X)[1]
Y = data[:, -1] #the last col, every row
Y = np.reshape(Y, (m, 1)) #dont know why this is needed, but it is
X = np.hstack([np.ones([m, 1]), X])

X_norm, mu, sigma = normalize(X, n)
theta, cost = gradient_descent_loop(X_norm, Y, m, n)
draw_cost(cost)

if n == 1: #provisional
    draw_points_plot(X_norm, Y, theta)
    A, B, Z = make_paint_data(X, Y)
    draw_cost_3D(A, B, Z)
    draw_contour(A, B, Z)

print("Values of theta: " + str(theta))
user_values = np.array(list(map(float, input("Enter query values: ").split())), dtype=float)
user_values = normalize_user_values(user_values, mu, sigma)
print("Your prediction: " + str(int(h(user_values, theta))))
