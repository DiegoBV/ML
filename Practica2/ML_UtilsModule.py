import numpy as np
from pandas.io.parsers import read_csv

class Data_Management:

    @staticmethod
    def load_csv(file_name):
        """
        Load the csv file. Returns numpy array
        """
        values = read_csv(file_name, header=None).values

        #always float
        return values.astype(float)

    @staticmethod
    def get_parameters_from_training_examples(training_examples):
        """
        Returns the needed parameters from the training examples, aka X, Y, m, n
        """
        X = training_examples[:, :-1] #every col but the last
        m = np.shape(X)[0] #number of training examples
        n = np.shape(X)[1] #number of attributes
        Y = training_examples[:, -1] #the last col, every row
        Y = np.reshape(Y, (m, 1)) #dont know why this is needed, but it is (needed for numpy operations)

        return X, Y, m, n

    @staticmethod
    def add_column_left_of_matrix(matrix):
        new_matrix = np.hstack([np.ones([np.shape(matrix)[0], 1]), matrix]) #convention in linear regr
        return new_matrix


class Normalization:
    """
    Regression Normalization (adding the one columns)
    """
    
    @staticmethod
    def normalize_data_matrix(X):
        """
        Normalize the given matrix using the mean and deviation of every column (attribute).
        Returns the normalized matrix, the mu vector (mean of every attribute) and sigma vector (deviation of every attribute).
        """
        n = np.shape(X)[1]
        X_norm = np.empty_like(X)
        sigma = np.empty(n)
        mu = np.empty(n)

        for i in range(0, n):
            mu[i] = np.mean(X[:, i]) #mean of every column or attribute
            sigma[i] = np.std(X[:, i]) #deviation of every column or attribute

            new_value = (X[:, i] - mu[i])/sigma[i] #normalize
            X_norm[:, i] = new_value

        return X_norm, mu, sigma

    @staticmethod
    def normalize_single_attributes(attributes, mu, sigma):
        """
        Normalize a single set of different attributes using mu and sigma vectors obtained in a
        previous normalization 
        """
        new_data = (attributes - mu[:])/sigma[:] #normalize
        return new_data

