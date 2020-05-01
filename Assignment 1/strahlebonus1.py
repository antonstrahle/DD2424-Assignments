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

def computeGradients(X, Y, W, b, lmda):
	
	nb = X.shape[1]
	
	p = evaluateClassifier(X, W, b)
	g = -(Y-p)
	
	gradW = np.matmul(g, X.transpose())/nb + 2*lmda*W
	gradB = np.matmul(g, np.ones((nb, 1)))/nb 
		
	return gradW, gradB

#Improvements (Bonus 1). Wanted to try object oriented

#Ensemble components

class mbGD(object):
	
	def __init__(self, intW, intB, xTrain = xTrain, yTrain = yTrain, xVal = xVal, lVal = lVal):
		self.W = intW
		self.b = intB
		self.xt = xTrain
		self.yt = yTrain
		self.xv = xVal
		self.lv = lVal
		
	def train(self, GDParams, lmda):
		
		bestW = self.W
		bestb = self.b
		
		nBatch = GDParams[0]
		eta = GDParams[1]
		nEpochs = GDParams[2]
		N = self.xt.shape[1]
		
		bestAcc = computeAccuracy(self.xv, self.lv, bestW, bestb)
		
		for i in range(nEpochs):
			
			p = np.random.permutation(N)
			
			permX = self.xt[:, p]
			permY = self.yt[:, p]

			
			for j in range(N//nBatch):
				jStart = (j-1)*nBatch
				jEnd = j*nBatch - 1
				XBatch = permX[:, jStart:jEnd]
				YBatch = permY[:, jStart:jEnd]
				
				gradW, gradb = computeGradients(XBatch, YBatch, self.W, self.b, lmda)
				
				self.W -= eta*gradW
				self.b -= eta*gradb
			
			
			acc = computeAccuracy(self.xv, self.lv, bestW, bestb)
			
			if acc > bestAcc:
				
				bestW = self.W
				bestb = self.b
			
		self.W = bestW
		self.b = bestb
		
	
	def classify(self, X):
		
		p = evaluateClassifier(X, self.W, self.b)
		
		return np.argmax(p, axis = 0)

#Ensemble Classifier		

class ensembleMBGD(object):

	def __init__(self, nNetworks):
		self.n = nNetworks
		self.classifiers = []
		
	def train(self, GDParams, lmda):
		
		for i in range(self.n):
			""" Other options? """
			W = np.random.normal(0, 0.01, (10,3072)) 
			b = np.random.normal(0, 0.01, (10,1))

			new = mbGD(W, b)
			new.train(GDParams, lmda)
		
			self.classifiers.append(new)
	
	def classify(self, X):
		
		ensembleOut = np.zeros((self.n, X.shape[1]))
		
		for i in range(self.n):
			out = self.classifiers[i].classify(X)
			ensembleOut[i,] = out
			
		return sc.mode(ensembleOut, axis = 0)[0]



tot = []

for i in range(10):

	ensemble = ensembleMBGD(5)

	ensemble.train([100, 0.001, 100], 0.1)

	opt = ensemble.classify(xTest)
		
	a = np.mean(np.equal(opt, lTest))
	
	tot.append(a)


print("Mean " + str(np.mean(tot)))
print("Std " + str(np.std(tot))) 



ensembel = ensembleMBGD(10)

ensembel.initialize([100, 0.001, 40], 0.1)

opt = ensembel.classify(xTest)
	
print(np.mean(np.equal(opt, lTest)))
	
	
#Bonus 2

def computeSVMLoss(X, Y, W, b, lmda):
	s = np.matmul(W,X) + b
	y = np.argmax(Y, axis = 0)
	
	l = 0
	
	for i in range(X.shape[1]):
		for j in range(Y.shape[0]):
			if j != y[i]:
				l += np.maximum(s[j,i]-s[y[i],i] + 1, 0)
	
	return l/X.shape[1] + lmda*np.sum(W**2)


def computeSVMGrads(X, Y, W, b, lmda):
	
	gradW = np.zeros(W.shape)
	gradb = np.zeros(b.shape)
	
	s = np.matmul(W,X) + b
	y = np.argmax(Y, axis = 0)
	
	"""tricky to get working using matrices"""
	
	for i in range(X.shape[1]):
		for j in range(Y.shape[0]):
			if j != y[i]:
				if s[j,i]-s[y[i],i] > -1:
				
					gradW[j] += X[:,i]
					gradb[j] += 1
					gradW[y[i]] -= X[:,i]
					gradb[y[i]] -= 1
					
	
	gradW = gradW/X.shape[1] + 2*lmda*W
	gradb /= X.shape[1]
	
	
	return gradW, gradb

def miniBatchSVMGD(X, Y, GDParams, intW, intb, lmda, xVal, yVal):
	
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
			
			gradW, gradb = computeSVMGrads(XBatch, YBatch, W, b, lmda)
			
			W -= eta*gradW
			b -= eta*gradb
		
		tc = computeSVMLoss(X, Y, W, b, lmda)
		vc = computeSVMLoss(xVal, yVal, W, b, lmda)
		
		trainingCost.append(tc)
		validationCost.append(vc)
	
	return W, b, trainingCost, validationCost


params = [100, 0.001, 40]

finalW, finalb, tc, vc = miniBatchSVMGD(xTrain, yTrain, params, W, b, 1, xVal, yVal)

plt.plot(tc, label = "Training Cost")
plt.plot(vc, label = "Validation Cost")
plt.legend(loc='best')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Cost', fontsize=16)
plt.show()

montage(finalW)

print("Final Accuracy: " + str(computeAccuracy(xVal, lVal, finalW, finalb)))

