from scipy.optimize import fmin_tnc as tnc
from ML_UtilsModule import Data_Management
from ML_UtilsModule import Normalization
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as pf
import numpy as np
import sys

learning_rate = 1.0
NUM_TRIES = 10

def polinomial_features(X, grado):
    poly = pf(grado)
    return poly, poly.fit_transform(X)

def J(theta, X, Y):
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    theta = np.reshape(theta, (1, n))
    var1 = np.dot(np.transpose((np.log(g(np.dot(X, np.transpose(theta)))))), Y)
    
    aux1 = np.dot(X, np.transpose(theta))
    aux2 = 1 - g(aux1)
    aux3 = np.log(aux2)
    
    var2 = np.dot(np.transpose(aux3), (1 - Y))
    var3 = (learning_rate/(2*m)) * np.sum(theta[1:]**2)
    return (((-1/m)*(var1 + var2)) + var3)

def gradient(theta, X, Y):
    m = np.shape(X)[0]
    n = np.shape(X)[1]
    theta = np.reshape(theta, (1, n))
    var1 = np.transpose(X)
    var2  = g(np.dot(X, np.transpose(theta)))-Y
    
    theta = np.c_[[0], theta[:, 1:]]
    var3 = (learning_rate/m) * theta
    return ((1/m) * np.dot(var1, var2)) + np.transpose(var3)

def g(z):
    """
    1/ 1 + e ^ (-0^T * x)
    """
    return 1/(1 + np.exp(-z))


def draw_data(X, Y):
    pos = np.where(Y == 0)[0] #vector with index of the Y = 1
    plt.scatter(X[pos, 0], X[pos, 1], marker='.', c='r', label = "Non Legendary")
    pos = np.where(Y == 1)[0].ravel() #vector with index of the Y = 1
    plt.scatter(X[pos, 0], X[pos, 1], marker='.', c='y', label = "Legendary")


def draw_decision_boundary(theta, X, Y, poly):
    x0_min, x0_max = X[:,0].min(), X[:,0].max()
    x1_min, x1_max = X[:,1].min(), X[:,1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x0_min, x0_max), np.linspace(x1_min, x1_max))
    
    sigm = g(poly.fit_transform(np.c_[ xx1.ravel(), xx2.ravel()]).dot(theta))
    sigm = sigm.reshape(xx1.shape)

    plt.contour(xx1, xx2, sigm, [0.5], linewidths = 1, colors = 'g')

def format_graphic (figure, ax, graphic_attr_names, score, polyDegree, sigma, mu):
    
    #formatting the graphic with some labels
    plt.xlabel(graphic_attr_names[0])
    plt.ylabel(graphic_attr_names[1])
    plt.suptitle(("Score: " + str(float("{0:.3f}".format(score))) + ", poly: " + str(polyDegree)))
    figure.legend()
    
    #set the labels to non-normalized values
    figure.canvas.draw()
    labels = [item for item in plt.xticks()[0]]
    for i in range(len(labels)):
        labels[i] = int(round((labels[i] * sigma[0, 0]) + mu[0, 0], -1))
    ax.xaxis.set_ticklabels(labels)

    labels = [item for item in plt.yticks()[0]]
    for i in range(len(labels)):
        labels[i] = int(round((labels[i] * sigma[0, 1]) + mu[0, 1], -1))
    ax.yaxis.set_ticklabels(labels)
    
def draw(theta, X, Y, poly, graphic_attr_names, score, polyDegree, sigma, mu):
    figure, ax = plt.subplots()
    
    draw_data(X, Y)
    draw_decision_boundary(theta, X, Y, poly)
    
    format_graphic(figure, ax, graphic_attr_names, score, polyDegree, sigma, mu)
    
    plt.show()

def checkLearned(X, Y, theta):     
    checker = g(np.dot(X, np.transpose(theta)))
    checker = np.around(checker)
    checker = np.reshape(checker, (np.shape(checker)[0], 1))
    truePositives = 0
    falsePositives = 0
    falseNegatives = 0
    
    for i in range(np.size(checker)):
        if checker[i] == 1 and y[i] == 1:
            truePositives += 1
        elif checker[i] == 1 and y[i] == 0:
            falsePositives += 1
        elif checker[i] == 0 and y[i] == 1:
            falseNegatives += 1    
    
    if truePositives == 0:
        return 0
        
    recall = (truePositives/(truePositives + falseNegatives)) 
    precision = (truePositives/(truePositives + falsePositives))
    score = 2 *(precision*recall/(precision + recall))

    return score

graphic_attr_names = ["capture_rate", "base_egg_steps"]
X, y = Data_Management.load_csv_svm("pokemon.csv", graphic_attr_names)
X, mu, sigma = Normalization.normalize_data_matrix(X)

X, y, trainX, trainY, validationX, validationY, testingX, testingY = Data_Management.divide_legendary_groups(X, y)

#------------------------------------------------------------------------------------------------- 

allMaxPercent = []
allMaxElev = []
allMaxPoly = []
allMaxThetas = []

Xused = validationX
Yused = validationY

for t in range(NUM_TRIES):
    i = 1
    polyMaxPercent = 0
    maxPercent = 0
    currentPercent = 0
    maxTh = None
    maxPoly = None
    
    poly, X_poly = polinomial_features(Xused, i)
    theta = np.zeros([1, np.shape(X_poly)[1]], dtype=float)
    
    theta = tnc(func=J, x0=theta, fprime=gradient, args=(X_poly, Yused))[0]
    
    currentPercent = checkLearned(X_poly, Yused, theta)

    maxTh = theta
    maxPoly = poly
    
    while currentPercent > maxPercent:
        maxPercent =  currentPercent
        polyMaxPercent = i
        maxTh = theta
        maxPoly = poly
        
        i = i + 1
        
        poly, X_poly = polinomial_features(Xused, i)
        theta = np.zeros([1, np.shape(X_poly)[1]], dtype=float)
        
        theta = tnc(func=J, x0=theta, fprime=gradient, args=(X_poly, Yused))[0]
        
        currentPercent = checkLearned(X_poly, Yused, theta)
        
    allMaxPercent.append(maxPercent)
    allMaxElev.append(polyMaxPercent)
    allMaxPoly.append(maxPoly)
    allMaxThetas.append(maxTh)
    
indx = allMaxPercent.index(max(allMaxPercent))

draw(allMaxThetas[indx], Xused, Yused, allMaxPoly[indx], graphic_attr_names, max(allMaxPercent), allMaxElev[indx],sigma, mu)

#------------------------------------------------------------------------------------------------- 