import numpy as np
from ML_UtilsModule import Data_Management
import sys

def h(x, _theta):
    return (np.dot(x, _theta)) #scalar product

def normal_equation(X, Y):
    inv = np.linalg.pinv((np.dot(np.transpose(X), X))) #(X^T * X)^-1
    transp = np.dot(np.transpose(X), Y) #X^T * Y
    return np.dot(inv, transp)

def adapt_user_values(user_values):
    user_values = np.hstack([[1], user_values])
    return user_values

data = Data_Management.load_csv(sys.argv[1])
X, Y, m, n = Data_Management.get_parameters_from_training_examples(data)
X = np.hstack([np.ones([m, 1]), X])

theta = normal_equation(X, Y)

#Print theta's values and ask for predictions
print("Values of theta: " + str(theta) + "\n")
while True:
    user_values = np.array(list(map(float, input("Enter query values: ").split())), dtype=float)
    if user_values.size == 0:
        break
    user_values = adapt_user_values(user_values)
    print("Your prediction: " + str(int(h(user_values, theta))) + "\n")
