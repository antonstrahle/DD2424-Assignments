import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.io as sio

def softMax(x):
	""" Standard definition of the softmax function """
	return np.exp(x) / np.sum(np.exp(x), axis=0)

def LoadBatch(filename):
	""" Copied from the dataset website """
	with open('Datasets/'+filename, 'rb') as fo:    
		dict = pickle.load(fo, encoding='bytes')    
	return dict

def ComputeGradsNum(X, Y, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));

	c = ComputeCost(X, Y, W, b, lamda);
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost(X, Y, W, b_try, lamda)
		grad_b[i] = (c2-c) / h

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost(X, Y, W_try, b, lamda)
			grad_W[i,j] = (c2-c) / h

	return [grad_W, grad_b]

def evaluateClassifier(X, W, b):
	s = np.matmul(W,X) + b
	return softMax(s)

def computeCost(X, Y, W, b, lmda):
	
	p = evaluateClassifier(X, W, b)
	
	return np.mean(-np.log(np.diag(np.matmul(Y.transpose(), p)))) + lmda*np.sum(np.square(W))

def ComputeGradsNumSlow(X, Y, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = computeCost(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2 = computeCost(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = computeCost(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = computeCost(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]

def montage(W):
	""" Display the image for each label in W """
	fig, ax = plt.subplots(1,10)
	for i in range(10):
		im  = W[i,:].reshape(32,32,3, order='F')
		sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
		sim = sim.transpose(1,0,2)
		ax[i].imshow(sim, interpolation='nearest')
		ax[i].set_title("y="+str(i))
		ax[i].axis('off')
	plt.show()
