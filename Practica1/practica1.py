from UtilsModule import Normalization
from UtilsModule import load_csv
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys

learning_rate = 0.01

def h(x, _theta):
    """
    H = O^T * X
    """
    return (np.dot(x, np.transpose(_theta))) #scalar product

def J(X, Y, _theta):
    """
    Cost function
    """
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

def make_paint_data(X, Y):
    """
    Slide's code
    """
    step= 0.1
    Theta0 = np.arange(-10, 10, step)
    Theta1 = np.arange(-1, 4, step)
    X_aux = np.hstack([np.ones([np.shape(X)[0], 1]), X])

    Theta0, Theta1 = np.meshgrid(Theta0, Theta1)

    coste = np.empty_like(Theta0)

    for ix, iy in np.ndindex(Theta0.shape):
        coste[ix, iy] = J(X_aux, Y, [Theta0[ix, iy], Theta1[ix, iy]])

    return Theta0, Theta1, coste

def draw_points_plot(X, Y, _theta):
    """
    Draw linear function with X points
    """
    plt.figure()
    plt.scatter(X[:, 1], Y, 1, "red")
    plt.plot(X[:, 1:], h(X, _theta), color="grey")
    plt.show()

def draw_cost_3D(X, Y, Z):
    """
    Draw 3D cost 
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d') 

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)  

    fig.colorbar(surf, shrink =0.5, aspect=5)
    plt.show()

def draw_contour(X, Y, Z):
    """
    Draw the contour
    """
    plt.figure()
    plt.contour(X, Y, Z)
    plt.show()

def draw_cost(cost):
    """
    Draw the linear progression of the cost
    """
    plt.figure()
    X = np.linspace(0, 400, len(cost))
    plt.plot(X, cost)
    plt.show()
    
def gradient_descent_loop(X, Y, m, n):
    """
    Gradient descent. Minimize till convergence
    """
    theta = np.zeros([1, n + 1], dtype=float)
    cost = np.array([], dtype=float)
    auxCost = sys.maxsize
    while True:
        minimize(X, Y, m, n, theta)
        cost = np.append(cost, J(X, Y, theta))
        if abs(auxCost - cost[-1]) < 1e-4:
            break #Stops the loop when we reach the convergence value of 10^-4
        auxCost = cost[-1]
    
    return theta, cost

data = load_csv(sys.argv[1])
X = data[:, :-1] #every col but the last
m = np.shape(X)[0] #number of training examples
n = np.shape(X)[1]
Y = data[:, -1] #the last col, every row
Y = np.reshape(Y, (m, 1)) #dont know why this is needed, but it is

X_norm, mu, sigma = Normalization.normalize_data_set(X)
theta, cost = gradient_descent_loop(X_norm, Y, m, n)
draw_cost(cost)

if n == 1: #provisional
    draw_points_plot(X_norm, Y, theta)
    A, B, Z = make_paint_data(X, Y)
    draw_cost_3D(A, B, Z)
    draw_contour(A, B, Z)

#Print theta's values and ask for predictions
print("Values of theta: " + str(theta))
while True:
    user_values = np.array(list(map(float, input("Enter query values: ").split())), dtype=float)
    if user_values.size == 0:
        break
    user_values = Normalization.normalize_single_attributes(user_values, mu, sigma)
    print("Your prediction: " + str(int(h(user_values, theta))) + "\n")
