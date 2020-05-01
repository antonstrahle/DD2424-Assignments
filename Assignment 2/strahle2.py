import numpy as np
import pandas as pd
import pickle
import math
import matplotlib.pyplot as plt
import scipy.stats as sc

#Problems due to too small values with np.exp


def softMax(x):

	return np.exp(x-np.amax(x))/np.sum(np.exp(x-np.amax(x)), axis = 0)

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
	
	mat = (xOut.transpose()-np.mean(xTrain, axis = 1))/np.std(xTrain, axis = 1)
		
	return mat.transpose()

#Normalized Data

xTrain = transformData(xTrain)
xVal = transformData(xVal)
xTest = transformData(xTest)

#Initializing parameters

def initializeParams(classes = 10, dataDim = 3072, hiddenLayers = 50):

	W1 = np.random.normal(0, 1/math.sqrt(dataDim), (hiddenLayers,dataDim))
	b1 = np.zeros((hiddenLayers,1))

	W2 = np.random.normal(0, 1/math.sqrt(hiddenLayers), (classes,hiddenLayers))
	b2 = np.zeros((classes,1))
	
	return W1, b1, W2, b2

W1, b1, W2, b2 = initializeParams()

#Evaluate Classifiers

def evaluateClassifier(X, W1, b1, W2, b2):
	
	h = np.matmul(W1,X) + b1
	h[h<0] = 0
	s = np.matmul(W2,h) + b2
	p = softMax(s)
	
	return p, h

#Compute Cost
"""
def computeCost(X, Y, W1, b1, W2, b2, lmda):
	
	p,_ = evaluateClassifier(X, W1, b1, W2, b2)
	l = np.sum(np.diag(-np.log(np.matmul(Y.transpose(), p))))/X.shape[1]
	j = l + lmda*(np.sum(W1**2) + np.sum(W2**2)) 
	
	return j
"""

#Alt Compute Cost to save memory

def computeCost(X, Y, W1, b1, W2, b2, lmda):
	
	N = X.shape[1]
	
	p,_ = evaluateClassifier(X, W1, b1, W2, b2)
	
	lab = np.argmax(Y, axis = 0)
	
	l = 0
	
	for i in range(N):
		
		l -= np.log(p[lab[i], i]) 
	
	j = l/N + lmda*(np.sum(W1**2) + np.sum(W2**2)) 
	
	return j


#Compute Accuracy

def computeAccuracy(X, y, W1, b1, W2, b2):
	p,_ = evaluateClassifier(X, W1, b1, W2, b2)
	opt = np.argmax(p, axis = 0) 
	return np.mean(np.equal(opt, y))

#Computes gradients for 2 layer NN.

def computeGradients(X, Y, W1, b1, W2, b2, lmda):
	nb = X.shape[1]
	
	p, h = evaluateClassifier(X, W1, b1, W2, b2)
	g1 = -(Y-p)
	
	gradW2 = np.matmul(g1,h.transpose())/nb
	gradB2 = np.matmul(g1,np.ones((nb, 1)))/nb
	
	g2 = np.matmul(W2.transpose(),g1)
	g3 = np.multiply(g2,h > 0)
	
	gradW1 = np.matmul(g3,X.transpose())/nb
	gradB1 = np.matmul(g3,np.ones((nb, 1)))/nb 
	
	gradW1 += 2*lmda*W1
	gradW2 += 2*lmda*W2
	
	return gradW1, gradB1, gradW2, gradB2


#Numerical Gradients

def computeGradientsNum(X, Y, W1, b1, W2, b2, lmda, h):
	
	gradW1 = np.zeros(W1.shape)
	gradB1 = np.zeros(b1.shape)
	
	gradW2 = np.zeros(W2.shape)
	gradB2 = np.zeros(b2.shape)
	
	#b1
	for i in range(len(b1)):
		b1Try = np.array(b1)
		b1Try[i] -= h
		c1 = computeCost(X, Y, W1, b1Try, W2, b2, lmda)

		b1Try = np.array(b1)
		b1Try[i] += h
		c2 = computeCost(X, Y, W1, b1Try, W2, b2, lmda)

		gradB1[i] = (c2-c1) / (2*h)
		
	#b2
	for i in range(len(b2)):
		b2Try = np.array(b2)
		b2Try[i] -= h
		c1 = computeCost(X, Y, W1, b1, W2, b2Try, lmda)

		b2Try = np.array(b2)
		b2Try[i] += h
		c2 = computeCost(X, Y, W1, b1, W2, b2Try, lmda)

		gradB2[i] = (c2-c1) / (2*h)
	
	#W1
	for i in range(W1.shape[0]):
		for j in range(W1.shape[1]):
			W1Try = np.array(W1)
			W1Try[i,j] -= h
			c1 = computeCost(X, Y, W1Try, b1, W2, b2, lmda)

			W1Try = np.array(W1)
			W1Try[i,j] += h
			c2 = computeCost(X, Y, W1Try, b1, W2, b2, lmda)

			gradW1[i,j] = (c2-c1) / (2*h)
	
	#W2
	for i in range(W2.shape[0]):
		for j in range(W2.shape[1]):
			W2Try = np.array(W2)
			W2Try[i,j] -= h
			c1 = computeCost(X, Y, W1, b1, W2Try, b2, lmda)

			W2Try = np.array(W2)
			W2Try[i,j] += h
			c2 = computeCost(X, Y, W1, b1, W2Try, b2, lmda)

			gradW2[i,j] = (c2-c1) / (2*h)
	
	return gradW1, gradB1, gradW2, gradB2

#Comparing Gradients: All are similar up to 6 decimals

"""
myW1, myB1, myW2, myB2 = computeGradients(xTrain[:,1:10], yTrain[:,1:10], W1, b1, W2, b2, 0)

numW1, numB1, numW2, numB2 = computeGradientsNum(xTrain[:,1:10], yTrain[:,1:10], W1, b1, W2, b2, 0, h = 1e-5)

print(np.testing.assert_array_almost_equal(myW1, numW1, decimal = 6))

print(np.testing.assert_array_almost_equal(myW2, numW2, decimal = 6))

print(np.testing.assert_array_almost_equal(myB1, numB1, decimal = 6))

print(np.testing.assert_array_almost_equal(myB2, numB2, decimal = 6))

#Minibatch GD for 2 layer NN
"""


def miniBatchGD(X, Y, GDParams, intW1, intb1, intW2, intb2, lmda, xVal, yVal):
	
	W1 = intW1
	b1 = intb1
	
	W2 = intW2
	b2 = intb2
	
	trainingCost = []
	validationCost = []
	
	nBatch = GDParams[0]
	etaMin = GDParams[1]
	etaMax = GDParams[2]
	nEpochs = GDParams[3]
	nS = GDParams[4]
	
	N = X.shape[1]
	
	eta = etaMin
	
	t = 0
	
	tc = computeCost(X, Y, W1, b1, W2, b2, lmda)
	vc = computeCost(xVal, yVal, W1, b1, W2, b2, lmda)
	
	trainingCost.append(tc)
	validationCost.append(vc)
	
	
	for i in range(nEpochs):
		
		p = np.random.permutation(N)
		
		permX = X[:, p]
		permY = Y[:, p]
		
		for j in range(N//nBatch):
			jStart = (j-1)*nBatch
			jEnd = j*nBatch - 1
			XBatch = permX[:, jStart:jEnd]
			YBatch = permY[:, jStart:jEnd]
			
			gradW1, gradb1, gradW2, gradb2 = computeGradients(XBatch, YBatch, W1, b1, W2, b2, lmda)
			
			W1 -= eta*gradW1
			b1 -= eta*gradb1
			W2 -= eta*gradW2
			b2 -= eta*gradb2
				
			if t <= nS:
				t +=1
				eta = etaMin + t * (etaMax - etaMin)/nS
			elif t < 2*nS:
				t +=1
				eta = etaMax - (t-nS) * (etaMax - etaMin)/nS
			else:
				eta = etaMin
				t = 0
			
		
		tc = computeCost(X, Y, W1, b1, W2, b2, lmda)
		vc = computeCost(xVal, yVal, W1, b1, W2, b2, lmda)
		
		trainingCost.append(tc)
		validationCost.append(vc)
		
		
	return W1, b1, W2, b2, trainingCost, validationCost


"""

#Single Training
params = [100, 1e-5, 1e-1, 10, 500]

finalW1, finalb1, finalW2, finalb2, tc, vc = miniBatchGD(xTrain, yTrain, params, W1, b1, W2, b2, 0.03, xVal, yVal)

plt.plot(tc, label = "Training Cost")
plt.plot(vc, label = "Validation Cost")
plt.legend(loc='best')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Cost', fontsize=16)
plt.show()

print("Final Accuracy: " + str(computeAccuracy(xTest, lTest, finalW1, finalb1, finalW2, finalb2)))

#Acc: 42%

"""

validationSize = 1000

xTrain1,yTrain1, lTrain1 = loadBatch('data_batch_1')
xTrain2,yTrain2, lTrain2 = loadBatch('data_batch_2')
xTrain3,yTrain3, lTrain3 = loadBatch('data_batch_3')
xTrain4,yTrain4, lTrain4 = loadBatch('data_batch_4')
xTrain5,yTrain5, lTrain5 = loadBatch('data_batch_5')

x = np.concatenate((xTrain1, xTrain2, xTrain3, xTrain4, xTrain5), axis=1)
y = np.concatenate((yTrain1, yTrain2, yTrain3, yTrain4, yTrain5), axis=1)
l = np.concatenate((lTrain1, lTrain2, lTrain3, lTrain4, lTrain5))

xTrainBigNN = x[:, :-validationSize]
yTrainBig = y[:, :-validationSize]
lTrainBig = l[:-validationSize]

xValBig = x[:, -validationSize:]
yValBig = y[:, -validationSize:]
lValBig = l[-validationSize:]

xTrainBig = transformData(xTrainBigNN, xTrainBigNN)
xValBig = transformData(xValBig, xTrainBigNN)

xTestBig, yTestBig, lTestBig = loadBatch('test_batch')
xTestBig = transformData(xTestBig, xTrainBigNN)

"""

nS = 1800 #2*45000/100

#Coarse

params = [100, 1e-5, 1e-1, 16, nS]

lmdaGrid = np.power(10,np.random.uniform(-2.75,-2.50,10))

for lmda in lmdaGrid:
	W1, b1, W2, b2 = initializeParams()
	finalW1, finalb1, finalW2, finalb2, _, _ = miniBatchGD(xTrainBig, yTrainBig, params, W1, b1, W2, b2, lmda, xValBig, yValBig)
	
	print("Final Accuracy for: " + str(computeAccuracy(xValBig, lValBig, finalW1, finalb1, finalW2, finalb2)))
	print("For: " + str(lmda))

"""	

nS = 1980 #4*49000/100
bestLmda = 0.00196

params = [100, 1e-5, 1e-1, 24, nS]

finalW1, finalb1, finalW2, finalb2, tc, vc = miniBatchGD(xTrainBig, yTrainBig, params, W1, b1, W2, b2, bestLmda, xValBig, yValBig)

plt.plot(tc, label = "Training Cost")
plt.plot(vc, label = "Validation Cost")
plt.legend(loc='best')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Cost', fontsize=16)
plt.show()

print("Final Accuracy for: " + str(computeAccuracy(xTestBig, lTestBig, finalW1, finalb1, finalW2, finalb2)))

