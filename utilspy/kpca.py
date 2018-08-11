import numpy as np
import math
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.metrics.pairwise import rbf_kernel

"""
Implementation of methods from the paper 
Kernel PCA and De-noising in feature spaces. 
Each function has a comment above it which contains
"(e)" where e denotes the corresponding equation from 
the paper. 
"""


def gaussianKernel(x, y, c):
    ''' Returns K(x,y) where K denotes gaussian kernel '''
    return math.exp(-(np.sqrt(np.dot(x - y, (x - y).conj())) ** 2) / c)


# return math.exp(-(np.linalg.norm(x-y)**2) / c)

def createK(data, c):
    ''' Returns K matrix containing inner products of the data using the kernel function
    so that K_ij := (phi(x_i)*phi(x_j)) '''
    return rbf_kernel(data, gamma=1 / c)


def createKOld(data, kernelFunction, c):
    ''' Returns K matrix containing inner products of the data using the kernel function
    so that K_ij := (phi(x_i)*phi(x_j)) '''
    return rbf_kernel(data, gamma=1 / c)


# l = len(data)
# K = np.zeros((l,l))
# for col in range(l):
#	for row in range(l):
#		K[row][col] = kernelFunction(data[row],data[col], c)
# return K

def calcBetaKOld(alphaK, data, x, c):
    ''' Returns the projection of x onto the eigenvector V_k '''
    BetaK = 0
    # print 'data.shape',data.shape
    # print 'x.shape', x.shape
    kernelVals = rbf_kernel(data, x.reshape(1, -1), 1 / c)
    for i, xi in enumerate(data):
        # BetaK += alphaK[i]*kernelFunction(xi,x,c)
        BetaK += alphaK[i] * kernelVals[i][0]
    return BetaK


def calcBetaK(alphaK, kernelVals):
    ''' Returns the projection of x onto the eigenvector V_k '''
    BetaK = 0
    BetaK = np.sum(alphaK * kernelVals)
    return BetaK


def centerK(K):
    ''' Returns centered K matrix, see K. Murphy 14.43 '''
    l = len(K)
    l_ones = np.ones((l, l), dtype=int) / l
    Kcentered = K - np.dot(l_ones, K) - np.dot(K, l_ones) + np.dot(l_ones, np.dot(K, l_ones))
    return Kcentered


def normAlpha(alpha, lambdas):
    ''' Returns new alpha corresponding to normalized eigen vectors,
    so that lambda_k(a^k * a^k) = 1 '''
    for i, a in enumerate(alpha):
        a /= np.sqrt(lambdas[i])
    return alpha


# def calcZold(alpha, data, x, kernelFunction, c,z0):
#	''' Equation (10), returns pre-image z for single input datapoint x '''
#	z = z0
#	iters=0
#	while iters <5:
#		numerator = 0
#		denom = 0
#		for i, xi in enumerate(data):
#			gammaI = calcGammaI(alpha, i, data, x, kernelFunction, c) * kernelFunction(z,xi,c)
#			numerator += gammaI * xi
#			denom += gammaI
#		z = numerator/denom
#		iters +=1
#	return z

def calcZWrapper(args):
    return calcZ(*args)


def calcZ(alpha, data, x, K, c, z0, idx):
    ''' Equation (10), returns pre-image z for single input datapoint x '''
    z = z0
    iters = 0
    maxIters = 10
    # calculate beta, gamma (do not change with each iteration)
    beta = [calcBetaKOld(aK, data, x, c) for aK in alpha]
    gamma = [calcGammaIOpt(alpha, i, beta) for i in range(len(data))]

    while iters < maxIters:  # iterate until convergence
        numerator = 0
        denom = 0
        k = rbf_kernel(data, z.reshape(1, -1), 1 / c)
        for i, xi in enumerate(data):
            gammaI = gamma[i] * k[i][0]
            numerator += gammaI * xi
            denom += gammaI
        if denom > 10 ** -12:  # handling numerical instability
            newZ = numerator / denom
            """
            if np.linalg.norm(z - newZ) < 10**-8: # convergence definition
                z = newZ
                break
            """
            z = newZ
            iters += 1
        else:
            # print "restarted point"
            iters = 0
            z = z0 + np.random.multivariate_normal(np.zeros(z0.size), np.identity(z0.size))
            numerator = 0
            denom = 0

    # print "iters:", iters
    return z


# def calcGammaI(alpha, i, data, x, kernelFunction, c):
#	''' returns gamma_i = sum_{k=1}^n Beta_k * alpha_i^k '''
#	gammaI = 0
#	alphaI = alpha.T[i]
#	for k, alphaKI in enumerate(alphaI):
#		gammaI += calcBetaK(alpha[k], kernelFunction, data, x, c) * alphaKI
#	return gammaI

def calcGammaIOpt(alpha, i, beta):
    ''' returns gamma_i = sum_{k=1}^n beta_k * alpha_i^k '''
    gammaI = 0
    alphaI = alpha.T[i]
    for k, alphaKI in enumerate(alphaI):
        gammaI += beta[k] * alphaKI
    return gammaI


def kernelPCADeNoise(kernelFunction, c, components, dataTrain, dataTest):
    Data = dataTrain

    l = len(Data)

    # build K
    # K = createK(Data, kernelFunction, c)
    K = createK(Data, c)

    # center K
    K = centerK(K)

    # find eigen vectors
    lLambda, alpha = np.linalg.eigh(K)  # (3)
    lambdas = lLambda / l  # /l with the notation from the paper (but not murphys)
    # drop negative and 0 eigenvalues and their vectors
    for i, l in enumerate(lambdas):
        if l > 10 ** (-8):
            lambdas = lambdas[i:]
            alpha = alpha[i:]
            break

    # use only the components largest eigenvalues with corresponding vectors
    lambdas = lambdas[-components:]
    alpha = alpha[-components:]

    # normalize alpha
    alpha = normAlpha(alpha, lambdas)

    # p=Pool()
    # Z = p.map(calcZWrapper, [(alpha, Data, x, K, c, x, i) for i, x in enumerate(dataTest)])

    Z = []
    for i in range(len(dataTest)):
        # print i
        Z.append(calcZ(alpha, Data, dataTest[i], K, c, dataTest[i], i))

    Z = np.array(Z)
    return Z


# if __name__ == '__main__':
#     # hyperparameters
#     c = 0.5
#
#     # For half-circle toy example
#     X, y = make_circles(n_samples=600, factor=.3, noise=.05)
#     X = np.array([x for i, x in enumerate(X) if x[1] > 0 and not y[i]])
#     Xtrain, Xtest = train_test_split(X, test_size=0.9)
#
#     Z = kernelPCADeNoise(gaussianKernel, c, 1, Xtrain, Xtest)
#
#     plt.plot(Xtrain.T[0], Xtrain.T[1], 'ro')
#     plt.plot(Z.T[0], Z.T[1], 'go')
#     plt.show()