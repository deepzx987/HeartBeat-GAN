import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
import scipy
import os
import seaborn as sns
# from tqdm import tqdm_notebook as tqdm
from keras.models import Model
from keras.layers import Input, Reshape, multiply, Embedding, merge, Concatenate
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, Conv1D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard
from joblib import Parallel, delayed
from keras import initializers
import keras
np.random.seed(10)
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
import time
import keras.backend as K
from scipy.special import kl_div
from scipy.spatial import distance
import keras.backend as K
from dtw import accelerated_dtw
from keras.optimizers import Adam, SGD

def reshape(X):
    if len(X.shape) == 1:
        X = X.reshape(X.shape[0], 1)
        return X
    else:
        if X.shape[-1] == 1:
            return X
        else:
            X = X.reshape(X.shape[0], X.shape[1], 1)
            return X

from keras.models import model_from_json
def load_model(data_dir, epoch=1, type='G', loss='binary_crossentropy', optim=Adam(0.0002, 0.5)):
    json_name = data_dir+str(epoch)+'_'+type+'.json'
    h5name = data_dir+str(epoch)+'_'+type+'.h5'
    # load json and create model
    json_file = open(json_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5name)
    # print("Loaded model from disk")
    # evaluate loaded model on test data
    loaded_model.compile(loss=loss, optimizer=optim)
    return loaded_model


# example of calculating the frechet inception distance
# calculate frechet inception distance
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_dtw(z_input, X):    
    dist_dtw = []
    manhattan_distance = lambda x, y: np.abs(x - y)
    for i in range(50):
        d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(z_input[i], X[i], dist=manhattan_distance)
        # print (d, end='\r')
        dist_dtw.append(d)
    dist_dtw = np.array(dist_dtw)
    return np.mean(dist_dtw)

def calculate_mmd(x1, x2):
    """
    maximum mean discrepancy (MMD) based on Gaussian kernel
    function for keras models (theano or tensorflow backend)
    
    - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
    Advances in neural information processing systems. 2007.
    """
    x1x1 = np.mean(np.exp(-1*np.square(x1 - x1)))
    x1x2 = np.mean(np.exp(-1*np.square(x1 - x2)))
    x2x2 = np.mean(np.exp(-1*np.square(x2 - x2)))
    return x1x1 - (2*x1x2) + x2x2


# def calculate_ED(temp_x,z_input): 
#     dist = (temp_x-z_input)**2
#     dist = np.mean(dist)
#     dist = np.sqrt(dist)
#     return dist

def calculate_PC(temp_x,z_input):
    val = []
    for i in range(temp_x.shape[0]):
        val.append(scipy.stats.pearsonr(temp_x[i],z_input[i])[0])
    val = np.array(val)
    val = np.mean(val)
    return val

from sklearn.metrics import mean_squared_error
def calculate_RMSE(temp_x,z_input):    
    return np.sqrt(mean_squared_error(temp_x,z_input))


def Dlp(A, B, p=2):
    cost = np.sum(np.power(np.abs(A - B), p))
    return np.power(cost, 1 / p)


def twed(A, timeSA, B, timeSB, nu=0.001, _lambda=0):
    # [distance, DP] = TWED( A, timeSA, B, timeSB, lambda, nu )
    # Compute Time Warp Edit Distance (TWED) for given time series A and B
    #
    # A      := Time series A (e.g. [ 10 2 30 4])
    # timeSA := Time stamp of time series A (e.g. 1:4)
    # B      := Time series B
    # timeSB := Time stamp of time series B
    # lambda := Penalty for deletion operation
    # nu     := Elasticity parameter - nu >=0 needed for distance measure
    # Reference :
    #    Marteau, P.; F. (2009). "Time Warp Edit Distance with Stiffness Adjustment for Time Series Matching".
    #    IEEE Transactions on Pattern Analysis and Machine Intelligence. 31 (2): 306–318. arXiv:cs/0703033
    #    http://people.irisa.fr/Pierre-Francois.Marteau/

    # Check if input arguments
    if len(A) != len(timeSA):
        print("The length of A is not equal length of timeSA")
        return None, None

    if len(B) != len(timeSB):
        print("The length of B is not equal length of timeSB")
        return None, None

    if nu < 0:
        print("nu is negative")
        return None, None

    # Add padding
    A = np.array([0] + list(A))
    timeSA = np.array([0] + list(timeSA))
    B = np.array([0] + list(B))
    timeSB = np.array([0] + list(timeSB))

    n = len(A)
    m = len(B)
    # Dynamical programming
    DP = np.zeros((n, m))

    # Initialize DP Matrix and set first row and column to infinity
    DP[0, :] = np.inf
    DP[:, 0] = np.inf
    DP[0, 0] = 0

    # Compute minimal cost
    for i in range(1, n):
        for j in range(1, m):
            # Calculate and save cost of various operations
            C = np.ones((3, 1)) * np.inf
            # Deletion in A
            C[0] = (
                DP[i - 1, j]
                + Dlp(A[i - 1], A[i])
                + nu * (timeSA[i] - timeSA[i - 1])
                + _lambda
            )
            # Deletion in B
            C[1] = (
                DP[i, j - 1]
                + Dlp(B[j - 1], B[j])
                + nu * (timeSB[j] - timeSB[j - 1])
                + _lambda
            )
            # Keep data points in both time series
            C[2] = (
                DP[i - 1, j - 1]
                + Dlp(A[i], B[j])
                + Dlp(A[i - 1], B[j - 1])
                + nu * (abs(timeSA[i] - timeSB[j]) + abs(timeSA[i - 1] - timeSB[j - 1]))
            )
            # Choose the operation with the minimal cost and update DP Matrix
            DP[i, j] = np.min(C)
    distance = DP[n - 1, m - 1]
    # return distance, DP
    return distance

def calculate_TWED(X, z_input, samples=10):
    Len = np.arange(X.shape[1])
    val = Parallel(n_jobs=-1)(delayed(twed)(X[i], Len, z_input[i], Len) for i in range(samples))
    return np.mean(np.array(val))

def evaluate(X, Z, n_classes, metric_to_calculate, n_batch, filename, samples=10):
    X = np.squeeze(X)
    Z = np.squeeze(Z)
    Win = (n_batch//3)
    results = []

    for classes in range(n_classes):
        temp_x = X[classes*Win:(classes+1)*Win]
        z_input = Z[classes*Win:(classes+1)*Win]
        for metrics in metric_to_calculate:
            if metrics == 'FID':
                results.append(calculate_fid(temp_x,z_input))
            if metrics == 'MMD':
                results.append(calculate_mmd(temp_x,z_input))
            if metrics == 'DTW':
                results.append(calculate_dtw(temp_x,z_input))
            if metrics == 'PC':
                results.append(calculate_PC(temp_x,z_input))
            if metrics == 'RMSE':
                results.append(calculate_RMSE(temp_x,z_input))
            if metrics == 'TWED':
                results.append(calculate_TWED(temp_x,z_input,samples=10))
    
    f = open(filename+'Stats.csv', 'a')
    for r in results:
        f.write(str(r)+',')
    f.write('\n')
    f.close()

    # return results

# def calculate_TWED(X, z_input, n_batch, samples=10):
#     Win = (n_batch//3)
#     X_new = np.vstack((X[:samples], X[Win:Win+samples], X[2*Win:2*Win+samples]))
#     z_input_new = np.vstack((z_input[:samples], z_input[Win:Win+samples], z_input[2*Win:2*Win+samples]))
#     Len = np.arange(X_new.shape[1])
#     val = Parallel(n_jobs=-1)(delayed(twed)(X_new[i], Len, z_input_new[i], Len) for i in range(X_new.shape[0]))
#     return np.mean(np.array(val))

# val = []
# R = X.shape[0]
# Len = np.arange(X.shape[1])
# for i in range(R):
#     val.append(twed(X[i], Len, z_input[i], Len))
# return np.mean(np.array(val))

# val = []
# for i in range(X.shape[0]):
#     val.append(twed(X[i], np.arange(X.shape[1]), z_input[i], np.arange(X.shape[1]), nu=0.001, _lambda=0))
# val = np.array(val)
# val = np.mean(val)

# def evaluate(temp_x, z_input, metric_to_calculate, n_batch, samples=10):
#     temp_x = np.squeeze(temp_x)
#     z_input = np.squeeze(z_input)
#     results = []
#     for j in metric_to_calculate:
#         if j == 'FID':
#             results.append(calculate_fid(temp_x,z_input))
#         if j == 'MMD':
#             results.append(calculate_mmd(temp_x,z_input))
#         if j == 'DTW':
#             results.append(calculate_dtw(temp_x,z_input))
#         # if j == 'ED':
#         #     results.append(calculate_ED(temp_x,z_input))
#         if j == 'PC':
#             results.append(calculate_PC(temp_x,z_input))
#         # if j == 'KLD':
#         #     results.append(np.mean(kl_div(temp_x,z_input)))
#         if j == 'RMSE':
#             results.append(calculate_RMSE(temp_x,z_input))
#         if j == 'TWED':
#             results.append(calculate_TWED(temp_x,z_input, n_batch, samples=10))
#     return results
