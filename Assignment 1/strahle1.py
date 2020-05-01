import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import scipy.stats as sc
from functions import *

def softMax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def loadBatch(filename):
	with open('../../Datasets/'+filename, 'rb') as fo:    
		dict = pickle.load(fo, encoding='bytes')
		
		y = np.asarray(dict[b'labels'])
		
		Y = np.zeros((10000,10))
		Y[np.arange(y.size), y] = 1

		X = np.asarray(dict[b'data'])/255
	
	return X.transpose(),Y.transpose(), y

xTrain,yTrain,lTrain = loadBatch("data_batch_1")

xVal,yVal,lVal = loadBatch("data_batch_2")

xTest,yTest,lTest = loadBatch("data_batch_3")


def transformData(xOut, xTrain = xTrain):
	return (xOut-np.mean(xTrain, axis = 0))/np.std(xTrain, axis = 0)

#Normalized Data

xTrain = transformData(xTrain)
xVal = transformData(xVal)
xTest = transformData(xTest)

#Initializing parameters

W = np.random.normal(0, 0.01, (10,3072))
b = np.random.normal(0, 0.01, (10,1))

#Evaluation

def evaluateClassifier(X, W, b):
	s = np.matmul(W,X) + b
	return softMax(s)
	
#p = EvaluateClassifier(xTrain[:, 1:100], W, b)

#Compute Cost

def computeCost(X, Y, W, b, lmda):
	
	p = evaluateClassifier(X, W, b)
	
	return np.mean(-np.log(np.diag(np.matmul(Y.transpose(), p)))) + lmda*np.sum(W**2)

#out = ComputeCost(xTrain[:, 1:100], yTrain[:,1:100],W, b, 0)

#Compute Accuracy

def computeAccuracy(X, y, W, b):
	p = evaluateClassifier(X, W, b)
	opt = np.argmax(p, axis = 0) 
	return np.mean(np.equal(opt, y))
	
#acc = computeAccuracy(xTrain, lTrain, W, b)

#Computing Batch Gradients

def computeGradients(X, Y, W, b, lmda):
	
	nb = X.shape[1]
	
	p = evaluateClassifier(X, W, b)
	g = -(Y-p)
	
	gradW = g@X.transpose()/nb + 2*lmda*W
	gradB = g@np.ones((nb, 1))/nb 
		
	return gradW, gradB
"""
myW, myB = computeGradients(xTrain[:,1:10], yTrain[:,1:10], W, b, 0)

numW, numB = ComputeGradsNumSlow(xTrain[:,1:10], yTrain[:,1:10], W, b, 0, h = 1e-6)

print(np.testing.assert_array_almost_equal(myW, numW, decimal = 6))
"""
#Basically no error for B, some for W

def miniBatchGD(X, Y, GDParams, intW, intb, lmda, xVal, yVal):
	
	W = intW
	b = intb
	
	trainingCost = []
	validationCost = []
	
	nBatch = GDParams[0]
	eta = GDParams[1]
	nEpochs = GDParams[2]
	N = X.shape[1]
	
	for i in range(nEpochs):
		
		p = np.random.permutation(N)
		
		permX = X[:, p]
		permY = Y[:, p]
		
		for j in range(N//nBatch):
			jStart = (j-1)*nBatch
			jEnd = j*nBatch - 1
			XBatch = permX[:, jStart:jEnd]
			YBatch = permY[:, jStart:jEnd]
			
			gradW, gradb = computeGradients(XBatch, YBatch, W, b, lmda)
			
			W -= eta*gradW
			b -= eta*gradb
		
		tc = computeCost(X, Y, W, b, lmda)
		vc = computeCost(xVal, yVal, W, b, lmda)
		
		trainingCost.append(tc)
		validationCost.append(vc)
	
	return W, b, trainingCost, validationCost

#Generating output

#1


params = [100, 0.1, 40]

finalW, finalb, tc, vc = miniBatchGD(xTrain, yTrain, params, W, b, 0, xVal, yVal)


plt.plot(tc, label = "Training Cost")
plt.plot(vc, label = "Validation Cost")
plt.legend(loc='best')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Cost', fontsize=16)
plt.show()

montage(finalW)


print("Final Accuracy: " + str(computeAccuracy(xVal, lVal, finalW, finalb)))

#2
"""
params = [100, 0.001, 40]

finalW, finalb, tc, vc = miniBatchGD(xTrain, yTrain, params, W, b, 0, xVal, yVal)


plt.plot(tc, label = "Training Cost")
plt.plot(vc, label = "Validation Cost")
plt.legend(loc='best')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Cost', fontsize=16)
plt.show()

#montage(finalW)

print("Final Accuracy: " + str(computeAccuracy(xVal, lVal, finalW, finalb)))

#With regularization

#3

params = [100, 0.001, 40]

finalW, finalb, tc, vc = miniBatchGD(xTrain, yTrain, params, W, b, 0.1, xVal, yVal)


plt.plot(tc, label = "Training Cost")
plt.plot(vc, label = "Validation Cost")
plt.legend(loc='best')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Cost', fontsize=16)
plt.show()

montage(finalW)


print("Final Accuracy: " + str(computeAccuracy(xVal, lVal, finalW, finalb)))


#4

params = [100, 0.001, 40]

finalW, finalb, tc, vc = miniBatchGD(xTrain, yTrain, params, W, b, 1, xVal, yVal)


plt.plot(tc, label = "Training Cost")
plt.plot(vc, label = "Validation Cost")
plt.legend(loc='best')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Cost', fontsize=16)
plt.show()

montage(finalW)

print("Final Accuracy: " + str(computeAccuracy(xVal, lVal, finalW, finalb)))
"""






