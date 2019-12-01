from matplotlib import pyplot as plt
from sklearn.svm import SVC
from scipy.io import loadmat
import numpy as np


def draw_decisition_boundary(X, y, svm):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    x1, x2 = np.meshgrid(x1, x2)
    yp = svm.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape)
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    plt.scatter(X[pos, 0], X[pos, 1], color='black', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], color='yellow',
    edgecolors='black', marker='o')
    plt.contour(x1, x2, yp)

def kernel_lineal():
    data = loadmat('ex6data1.mat')
    X, y = data['X'], data['y']

    svm = SVC(kernel='linear', C=1.0)
    svm.fit(X, y.ravel())

    plt.figure()
    draw_decisition_boundary(X, y, svm)
    plt.show()

    svm = SVC(kernel='linear', C=100)
    svm.fit(X, y.ravel())

    plt.figure()
    draw_decisition_boundary(X, y, svm)
    plt.show()

def kernel_gaussiano():
    data = loadmat('ex6data2.mat')
    X, y = data['X'], data['y']
    sigma = 0.1

    svm = SVC(kernel = 'rbf' , C= 1, gamma= (1 / ( 2 * sigma ** 2)) )
    svm.fit(X, y.ravel())

    plt.figure()
    draw_decisition_boundary(X, y, svm)
    plt.show()

def eleccion_parametros_C_y_Sigma():
    data = loadmat('ex6data3.mat')
    X, y, Xval, yval = data['X'], data['y'], data['Xval'], data['yval']
    possible_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    C_value, sigma = 0, 0
    max_score = 0
    best_svm = None

    for i in range(len(possible_values)):
        C_value = possible_values[i]
        for j in range(len(possible_values)):
            sigma = possible_values[j]
            svm = SVC(kernel = 'rbf' , C = C_value, gamma= (1 / ( 2 * sigma ** 2)))
            svm.fit(X, y.ravel())
            current_score = svm.score(Xval, yval) #calcula el score con los ejemplos de validacion (mayor score, mejor es el svm)
            if current_score > max_score:
                max_score = current_score
                best_svm = svm
    
    plt.figure()
    draw_decisition_boundary(X, y, best_svm)
    plt.show()

kernel_lineal()
kernel_gaussiano()
eleccion_parametros_C_y_Sigma()