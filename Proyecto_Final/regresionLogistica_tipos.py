from scipy.optimize import fmin_tnc as tnc
from ML_UtilsModule import Data_Management
from ML_UtilsModule import Normalization
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures as pf
import numpy as np
import sys

learning_rate = 0.1
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

def transform_y(y, num_etiquetas):
    y = np.reshape(y, (np.shape(y)[0], 1))
    mask = np.empty((num_etiquetas, np.shape(y)[0]), dtype=bool)
    for i in range( num_etiquetas):
        mask[i, :] = (y[:, 0] == i)

    mask = mask * 1

    return np.transpose(mask)

def des_transform_y(y, num_etiquetas):
    deTransY = np.where(y == 1)
    return deTransY[1]

def paint_pkmTypes(X, y, types = None, paintAll = False):
    '''
    Creates plt figure and draws the pkms used as input by the first two values of X
    and changes the color of them depending on the type

    X = attributes
    y = list of types, deTransformed (from 0 to 17)
    types = types that are going to be printed
    '''
    typesIndx = []
    typesNotSelectedIndx = []
    
    if(types == None):
        types = Data_Management.types_
        # esto es para que meta los indices del 0 al 17, de los tipos completos, como no lo conseguia, he pasado
        typesIndx = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
    else:
        for t in range(len(types)):
            typesIndx.append(Data_Management.types_.index(types[t]))


    colors = Data_Management.colors_

    if paintAll:
        plt.scatter(X[:, 0], X[:, 1], color=colors[18], marker='.', label = 'other')
                
    for i in range(len(typesIndx)):
        pos = (y == typesIndx[i]).ravel()
        plt.scatter(X[pos, 0], X[pos, 1], color='#ff0303', marker='+', label = types[i])


def draw_decision_boundary(theta, X, Y, poly, indx = 0):
    x0_min, x0_max = X[:,0].min(), X[:,0].max()
    x1_min, x1_max = X[:,1].min(), X[:,1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x0_min, x0_max), np.linspace(x1_min, x1_max))
    
    sigm = g(poly.fit_transform(np.c_[ xx1.ravel(), xx2.ravel()]).dot(theta))
    sigm = sigm.reshape(xx1.shape)

    plt.contour(xx1, xx2, sigm, [0.5], linewidths = 1, colors = Data_Management.colors_[indx])

def format_graphic (figure, ax, graphic_attr_names, score, polyDegree, sigma, mu):
    
    #formatting the graphic with some labels
    plt.xlabel(graphic_attr_names[0])
    plt.ylabel(graphic_attr_names[1])
    
    if polyDegree == 0:
        polyDegree = 1
        
    plt.suptitle(("Score: " + str(float("{0:.3f}".format(score))) + ", poly: " + str(polyDegree)))
    figure.legend()
    
    #set the labels to non-normalized values
    figure.canvas.draw()
    labels = [item for item in plt.xticks()[0]]
    for i in range(len(labels)):
        labels[i] = (round((labels[i] * sigma[0, 0]) + mu[0, 0], 1)) #int
    ax.xaxis.set_ticklabels(labels)

    labels = [item for item in plt.yticks()[0]]
    for i in range(len(labels)):
        labels[i] = (round((labels[i] * sigma[0, 1]) + mu[0, 1], 1)) #int
    ax.yaxis.set_ticklabels(labels)
    
def draw(theta, X, Y, poly, graphic_attr_names, score, polyDegree, sigma, mu, types = None, paintAll = False):
    figure, ax = plt.subplots()
    
    paint_pkmTypes(X, Y, types, paintAll)
    
    if(types == None):
        for i in range(18):
            draw_decision_boundary(theta[i], X, Y, poly, i)
    else:
        for t in range(len(types)):
            draw_decision_boundary(theta[Data_Management.types_.index(types[t])], X, Y, poly, Data_Management.types_.index(types[t]))
    
    format_graphic(figure, ax, graphic_attr_names, score, polyDegree, sigma, mu)
    
    plt.show()

def checkLearned(X, Y, theta):     
    checker = g(np.dot(X, np.transpose(theta)))
    maxIndexV = np.argmax(checker, axis = 1)
    checker = (Y[:] == maxIndexV)

    truePositives = 0
    falsePositives = 0
    falseNegatives = 0

    for i in range(np.shape(Y)[0]):
        for j in range(18):
            if maxIndexV[i] == j and y[i] == j:
                truePositives += 1
                break
            elif maxIndexV[i] == j and y[i] != j:
                falsePositives += 1
                break
            elif maxIndexV[i] != j and y[i] == j:
                falseNegatives += 1
                break

    if truePositives == 0:
        return 0

    recall = (truePositives/(truePositives + falseNegatives))
    precision = (truePositives/(truePositives + falsePositives))
    score = 2 *(precision*recall/(precision + recall))

    return score

def checkLearnedByType(X, Y, theta, indxTypeChecked):
    checker = g(np.dot(X, np.transpose(theta)))
    maxIndexV = np.argmax(checker, axis = 1)
    checker = (Y[:] == maxIndexV)

    truePositives = 0
    falsePositives = 0
    falseNegatives = 0
    
    
    for i in range(np.shape(Y)[0]):
            if maxIndexV[i] == indxTypeChecked and y[i] == indxTypeChecked:
                truePositives += 1
                break
            elif maxIndexV[i] == indxTypeChecked and y[i] != indxTypeChecked:
                falsePositives += 1
                break
            elif maxIndexV[i] != indxTypeChecked and y[i] == indxTypeChecked:
                falseNegatives += 1
                break

    if truePositives == 0:
        return 0

    recall = (truePositives/(truePositives + falseNegatives))
    precision = (truePositives/(truePositives + falsePositives))
    score = 2 *(precision*recall/(precision + recall))

    return score
    

graphic_attr_names = ["against_normal", "against_ice"]
types_rendered = ["steel"]
num_tipos = 18
X, y = Data_Management.load_csv_types_features("pokemon.csv", graphic_attr_names)
X, mu, sigma = Normalization.normalize_data_matrix(X)

X, y, trainX, trainY, validationX, validationY, testingX, testingY = Data_Management.divide_legendary_groups(X, y)

#------------------------------------------------------------------------------------------------- 
allMaxPercent = []
allMaxElev = []
allMaxPoly = []
allMaxThetas = []

Xused = X
Yused = y
Yused = transform_y(Yused, num_tipos)

for t in range(NUM_TRIES):
    i = 1
    polyMaxPercent = 0
    maxPercent = 0
    currentPercent = 0
    maxTh = None
    maxPoly = None
    
    poly, X_poly = polinomial_features(Xused, i)
    theta = np.zeros([num_tipos, np.shape(X_poly)[1]], dtype=float)
    
    for j in range(num_tipos):
        yTipo = np.reshape(Yused[:,j], (np.shape(Yused)[0], 1))
        theta[j] = tnc(func=J, x0=theta[j], fprime=gradient, args=(X_poly, yTipo))[0]
    
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
        theta = np.zeros([num_tipos, np.shape(X_poly)[1]], dtype=float)
        
        for j in range(num_tipos):
            yTipo = np.reshape(Yused[:,j], (np.shape(Yused)[0], 1))
            theta[j] = tnc(func=J, x0=theta[j], fprime=gradient, args=(X_poly, yTipo))[0]
        
        currentPercent = checkLearned(X_poly, Yused, theta)
        
    allMaxPercent.append(maxPercent)
    allMaxElev.append(polyMaxPercent)
    allMaxPoly.append(maxPoly)
    allMaxThetas.append(maxTh)
    
indx = allMaxPercent.index(max(allMaxPercent))
Yused = des_transform_y(Yused, num_tipos)

for pkm in range(18):
    draw(allMaxThetas[indx], Xused, Yused, allMaxPoly[indx], graphic_attr_names, max(allMaxPercent), allMaxElev[indx],sigma, mu, [Data_Management.types_[pkm]], True)

#------------------------------------------------------------------------------------------------- 