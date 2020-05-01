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

def initializeParams(classes = 10, dataDim = 3072, hiddenLayers = 300):

	W1 = np.random.normal(0, 1/math.sqrt(dataDim), (hiddenLayers,dataDim))
	b1 = np.zeros((hiddenLayers,1))

	W2 = np.random.normal(0, 1/math.sqrt(hiddenLayers), (classes,hiddenLayers))
	b2 = np.zeros((classes,1))
	
	return W1, b1, W2, b2

W1, b1, W2, b2 = initializeParams()

#Evaluate Classifiers

def evaluateClassifier(X, W1, b1, W2, b2, pDropout = 1):
	
	h = np.matmul(W1,X) + b1
	h[h<0] = 0
	
	dp1 = (np.random.uniform(0, 1, (h.shape)) < pDropout)/pDropout
	
	h = np.multiply(h, dp1)
	s = np.matmul(W2,h) + b2
	
	dp2 = (np.random.uniform(0, 1, (s.shape)) < pDropout)/pDropout
	
	s = np.multiply(s, dp2)
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

#Alt Compute Cost to avoid memory overload due to 49k x 49k matrix

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

"""
def emClassify(X, nws):
	
	ensembleOut = np.zeros((len(nws), X.shape[1]))
	
	for i in range(len(nws)):
		
		nw = nws[i]
		
		W1 = nw[0]
		b1 = nw[1]
		W2 = nw[2]
		b2 = nw[3]
		
		p, _ = evaluateClassifier(X, W1, b1, W2, b2)
		opt = np.argmax(p, axis = 0)
		ensembleOut[i,] = opt
	
	return sc.mode(ensembleOut, axis = 0)[0]
"""	
	

#Computes gradients for 2 layer NN.

def computeGradients(X, Y, W1, b1, W2, b2, lmda, pDropout):
	nb = X.shape[1]
	
	p, h = evaluateClassifier(X, W1, b1, W2, b2, pDropout)
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


#Minibatch GD for 2 layer NN



def miniBatchGD(X, Y, GDParams, intW1, intb1, intW2, intb2, lmda, xVal, lVal, pDropout):
	
	W1 = intW1
	b1 = intb1
	
	W2 = intW2
	b2 = intb2
	
	vacc = []
	tacc = []
	
	lTrain = np.argmax(Y, axis = 0) 
	
	nBatch = GDParams[0]
	etaMin = GDParams[1]
	etaMax = GDParams[2]
	nEpochs = GDParams[3]
	nS = GDParams[4]
	
	N = X.shape[1]
	
	eta = etaMin
	"""
	vc = computeAccuracy(xVal, lVal, W1, b1, W2, b2)
	tc = computeAccuracy(X, lTrain, W1, b1, W2, b2)
	
	vacc.append(vc)
	tacc.append(tc)
	"""
	
	t = 0
	
	for i in range(nEpochs):
		
		p = np.random.permutation(N)
		
		permX = X[:, p]
		permY = Y[:, p]
		
		for j in range(N//nBatch):
			jStart = (j-1)*nBatch
			jEnd = j*nBatch - 1
			XBatch = permX[:, jStart:jEnd]
			YBatch = permY[:, jStart:jEnd]
			
			gradW1, gradb1, gradW2, gradb2 = computeGradients(XBatch, YBatch, W1, b1, W2, b2, lmda, pDropout)
			
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
		"""
		vc = computeAccuracy(xVal, lVal, W1, b1, W2, b2)
		tc = computeAccuracy(X, lTrain, W1, b1, W2, b2)
		
		vacc.append(vc)
		tacc.append(tc)
		"""

	return W1, b1, W2, b2, vacc, tacc

#More data

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


nS = 490*5 #4*45000/100
pDropout = 0.80
params = [100, 1e-5, 1e-1, 80, nS]
lmda = 0.00196

finalW1, finalb1, finalW2, finalb2, _, _ = miniBatchGD(xTrainBig, yTrainBig, params, W1, b1, W2, b2, lmda, xValBig, yValBig, pDropout)

print("Final V Accuracy for: " + str(computeAccuracy(xValBig, lValBig, finalW1, finalb1, finalW2, finalb2)))
print("Final Training Accuracy for: " + str(computeAccuracy(xTrainBig, lTrainBig, finalW1, finalb1, finalW2, finalb2)))
print("Final Test Accuracy for: " + str(computeAccuracy(xTestBig, lTestBig, finalW1, finalb1, finalW2, finalb2)))

