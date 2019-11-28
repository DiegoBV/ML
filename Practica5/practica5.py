from ML_UtilsModule import Data_Management
from ML_UtilsModule import Normalization
from scipy.optimize import minimize as sciMin
from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np

_lambda = 0

def h(x, _theta):
    """
    H = O^T * X
    """
    return (np.dot(x, np.transpose(_theta))) #scalar product

def error_hipotesis(_theta, X, y):
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    _theta = np.reshape(_theta, (1, n))

    diff = (h(X, _theta) - y)**2

    return np.sum(diff) / (2 * m)


def J(_theta, X, Y):
    """
    Cost function
    """
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    
    _theta = np.reshape(_theta, (1, n))

    diff = (h(X, _theta) - Y)**2
    cost = np.sum(diff)/(2*m)
    cost += (_lambda * (np.sum(_theta**2)) / (2 * len(Y)))
    return cost

    # m = np.shape(X)[0]
    # aux = (_lambda/(2*m)) * np.sum(_theta**2)
    # print("cost " + str(1/(2*m) * np.sum((h(X, _theta) - Y)**2) + aux))
    # return (1/(2*m)) * np.sum((h(X, _theta) - Y)**2) + aux

def gradient(_theta, X, y):
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    
    _theta = np.reshape(_theta, (1, n))

    var1 = h(X, _theta) - y
    var2 = (_lambda/m) * _theta #thetha[0] = 0

    var3 = (1/m) * (np.dot(np.transpose(var1), X)) + var2
    return np.transpose(var3)

def pesos_aleat(L_in, L_out):
    pesos = np.random.uniform(-0.12, 0.12, (L_out, 1+L_in))
    
    return pesos

def generate_polynom_data(X, p):
    newMatrix = np.zeros((np.shape(X)[0], p))
    grades = np.arange(1, p + 1)
    newMatrix = X ** grades
    return newMatrix

def normalize_matrix(X):
    return Normalization.normalize_data_matrix(X)

def minimizar(theta, X, y):
    return J(theta, X, y), gradient(theta, X, y)

def draw_points_plot(X, Y, _theta):
    """
    Draw linear function with X points
    """
    plt.figure()
    plt.scatter(X[:, 1], Y, 20,marker='$F$',color= "red")
    kk = h(X, _theta)
    plt.plot(X[:, 1:], h(X, _theta), color="grey")
    plt.show()

def draw_plot(X, Y):    
    plt.plot(X, Y)
    

data = loadmat('ex5data1.mat')
X, y, Xval, yval, Xtest, ytest = data['X'], data['y'],  data['Xval'], data['yval'], data['Xtest'], data['ytest']
XPoly = generate_polynom_data(X, 2)
XPoly, mu, sigma = normalize_matrix(XPoly)
XPoly = Data_Management.add_column_left_of_matrix(XPoly)

theta = np.ones(XPoly.shape[1], dtype=float)
theta_min = sciMin(fun=minimizar, x0=theta,
 args=(XPoly, y),
 method='TNC', jac=True,
 options={'maxiter': 70}).x

draw_points_plot(XPoly, y, theta_min)
#---------------------------Parte 2 -------------------------------------
# X_transformed = Data_Management.add_column_left_of_matrix(X)
# Xval_transformed = Data_Management.add_column_left_of_matrix(Xval)

# error_array = np.array([], dtype=float)
# thetas = np.array([], dtype=float)
# error_array_val = np.array([], dtype=float)
# for i in range(1, np.shape(X_transformed)[0]):
#     theta = np.ones(X_transformed.shape[1], dtype=float)
    
#     theta_min = sciMin(fun=minimizar, x0=theta,
#     args=(X_transformed[0:  i], y[0: i]),
#     method='TNC', jac=True,
#     options={'maxiter': 70}).x
    
#     error_array = np.append(error_array, error_hipotesis(theta_min, X_transformed[0:  i], y[0: i]))
#     error_array_val = np.append(error_array_val, error_hipotesis(theta_min, Xval_transformed, yval))
#     thetas = np.append(thetas, theta_min)

# plt.figure()
# draw_plot(np.linspace(0, np.shape(X_transformed)[0], len(error_array)), error_array)
# draw_plot(np.linspace(0, np.shape(Xval_transformed)[0], len(error_array_val)), error_array_val)
# plt.show()

# # theta = np.ones(X_transformed.shape[1], dtype=float)
# # theta_min = sciMin(fun=minimizar, x0=theta,
# #  args=(X_transformed, y),
# #  method='TNC', jac=True,
# #  options={'maxiter': 70}).x

# draw_points_plot(X_transformed, y, theta_min)

