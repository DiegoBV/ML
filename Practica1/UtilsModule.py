import numpy as np
from pandas.io.parsers import read_csv

def load_csv(file_name):
    """
    Load the csv file. Returns numpy array
    """

    values = read_csv(file_name, header=None).values

    #always float
    return values.astype(float)

class Normalization:
    """
    Simple Normalization
    """
    
    @staticmethod
    def normalize_data_set(X):
        """
        Normalize the given matrix (data set) using the mean and deviation of every column (attribute).
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

        X_norm = np.hstack([np.ones([np.shape(X)[0], 1]), X_norm]) #convention in linear regr

        return X_norm, mu, sigma

    @staticmethod
    def normalize_single_attributes(attributes, mu, sigma):
        """
        Normalize a single set of different attributes using mu and sigma vectors obtained in a
        previous normalization 
        """
        new_data = (attributes - mu[:])/sigma[:] #normalize
        new_data = np.hstack([[1], new_data]) #convention in linear regr
        return new_data

