import numpy as np
import pandas as pd
import pickle
import math
import matplotlib.pyplot as plt
import scipy.stats as sc


def load(fn):
	
	content = open("../../TextData/"+fn, "r", encoding='utf8').read()
	chars = list(set(content))
	
	toChar = sorted(chars, key=str.lower)
	
	toInd = {}
	i = 0
	for char in toChar:
		toInd[char] = i
		i += 1
	
	return content, chars, toChar, toInd

class RNN(object):
	 
	def __init__(self, eta = 0.1, seqLength = 25, m = 100, sig = 0.01, fn = "goblet_book.txt"):
		
		self.data, self.chars, self.toChar, self.toInd = load(fn)
		self.nChars = len(self.chars)
		self.eta = eta
		self.seqLength = seqLength
		self.m = m
		self.b = np.zeros((m, 1))
		self.c = np.zeros((len(self.chars), 1))
		self.U = np.random.normal(0, sig, (m, len(self.chars)))
		self.W = np.random.normal(0, sig, (m, m))
		self.V = np.random.normal(0, sig, (len(self.chars), m))
	 
	def softMax(self, x):

		return np.exp(x-np.amax(x))/np.sum(np.exp(x-np.amax(x)), axis = 0)
	
	def evaluateClassifier(self, x, h):
		
		a = np.matmul(self.W, h) + np.matmul(self.U, x) + self.b
		h = np.tanh(a)
		o = np.matmul(self.V, h) + self.c
		p = self.softMax(o)
		
		return p, o, h, a
	
	def synthesize(self, h, sx, N):
		
		out = ""
		
		#Creating one-hot vector
		x = np.zeros((self.nChars, 1))
		x[sx] = 1
		
		for i in range(N):
			
			p,_,h,_ = self.evaluateClassifier(x, h)
			
			x = np.zeros((self.nChars, 1))
			xind = np.random.choice(range(self.nChars), p = p.flat)
			x[xind] = 1
			
			out += self.toChar[xind]
				
		print(out)
	
	def forward(self, X, Y, ph):
		
		l = 0
		x, p, o, h, a = {}, {}, {}, {}, {}
		h[-1] = ph 
		
		for i in range(len(X)):
			x[i] = np.zeros((self.nChars, 1))
			x[i][X[i]] = 1
			
			#Note that we have set h[-1] to accomodate for i = 1
			p[i], o[i], h[i], a[i] = self.evaluateClassifier(x[i], h[i-1])
			
			l -= float(np.log(p[i][Y[i]]))
		
		return l, p, o, h, a, x
	
	def computeGradients(self, X, Y, ph):
		
		#Forward
		l, p, o, h, a, x = self.forward(X, Y, ph)
		
		#Backward
		grads = {"W": np.zeros_like(self.W), "U": np.zeros_like(self.U), "V": np.zeros_like(self.V),
				 "b": np.zeros_like(self.b), "c": np.zeros_like(self.c)}
		
		temp = {"h": np.zeros_like(h[0]), "hn": np.zeros_like(h[0]),
				"a": np.zeros_like(a[0]), "o": np.zeros_like(p[0])}
		
		for i in range(len(X)-1, -1, -1):
			#Temporary grads used for the calculation of the grads of importance
			temp["o"] = p[i]
			temp["o"][Y[i]] -= 1
			temp["h"] = np.matmul(self.V.transpose(), temp["o"]) + temp["hn"]
			temp["a"] = np.multiply(temp["h"], 1 - np.square(np.tanh(a[i])))
			temp["hn"] = np.matmul(self.W.transpose(), temp["a"])
			
			#Grads of importance used in order to update parameters
			grads["V"] += np.matmul(temp["o"], h[i].transpose())
			grads["U"] += np.matmul(temp["a"], x[i].transpose())
			grads["W"] += np.matmul(temp["a"], h[i-1].transpose())
			grads["c"] += temp["o"]
			grads["b"] += temp["a"]
		
		return l, grads, h[len(X)-1]
	
	def train(self, itr = 100000):
		
		params = {"V": self.V, "U": self.U, "W": self.W, "c": self.c, "b":self.b}
		
		ada = {"W": np.zeros_like(self.W), "U": np.zeros_like(self.U), "V": np.zeros_like(self.V),
			   "b": np.zeros_like(self.b), "c": np.zeros_like(self.c)}
		
		h = np.zeros((self.m, 1))
		
		sll = []
		
		start = 0
		
		for i in range(itr):
			if (start+1)*self.seqLength > len(self.data):
				print("Starting Over \n")
				start = 0
			
			x = [self.toInd[char] for char in self.data[start*self.seqLength:(start+1)*self.seqLength]]
			y = [self.toInd[char] for char in self.data[(start*self.seqLength + 1):((start+1)*self.seqLength + 1)]]
			start += 1
			
			l, grads, h = self.computeGradients(x, y, h)
			
			for grad in grads:
				grads[grad] = np.clip(grads[grad], -5, 5) #faster than max(-5, min(grad, 5)) accoridng to documentation
			
			for param in params:
				ada[param] += np.square(grads[param])
				params[param] -= self.eta*grads[param]/np.sqrt(ada[param] + np.finfo(float).eps)
		
			if i == 0:
				sl = l
				bl = l
				
			else:
				sl = 0.999 * sl + 0.001 * l
				#Best parameters
				if l < bl:
					b = self.b
					c = self.c
					U = self.U 
					W = self.W 
					V = self.V
					h = h
			
			sll.append(sl)
			
			if i % 10000 == 0:
				print("After " + str(i) + " iterations")
				self.synthesize(h, 26, 200)
				print("")
		
		return sll, b, c, U, W, V, h
			
	def computeGradientsNum(self, X, Y, ph, h = 1e-4):
		
		params = {"V": self.V, "U": self.U, "W": self.W, "c": self.c, "b":self.b}
		grads = {"W": np.zeros_like(self.W), "U": np.zeros_like(self.U), "V": np.zeros_like(self.V),
				 "b": np.zeros_like(self.b), "c": np.zeros_like(self.c)}
		
		for param in params:
			for i in range(len(params[param].flat)):
				old = params[param].flat[i] #Save baseline
				
				params[param].flat[i] = old - h
				c1,_,_,_,_,_ = self.forward(X, Y, ph)
				
				params[param].flat[i] = old + h
				c2,_,_,_,_,_ = self.forward(X, Y, ph)
				
				params[param].flat[i] = old #Return baseline to not interfere with other gradients
				grads[param].flat[i] = (c2-c1)/(2*h) #Update
		
		return grads
		
"""

#Gradient Checking
test = RNN(m = 5)

start = 1
x = [test.toInd[char] for char in test.data[0:5]]
y = [test.toInd[char] for char in test.data[1:6]]

ph = np.zeros((test.m, 1))

_,myGrads,_ = test.computeGradients(x, y, ph)

numGrads = test.computeGradientsNum(x, y, ph)

for grad in myGrads:
	print("For: " + str(grad))
	np.testing.assert_array_almost_equal(myGrads[grad], numGrads[grad], decimal = 10)
	
#All equal up to 5 decimal places (some are equal up to more but the minimum was 5 for U)	

"""

rnn = RNN()

sll, rnn.b, rnn.c, rnn.U, rnn.W, rnn.V, h = rnn.train()

print("Final \n")

rnn.synthesize(h, 26, 1000)

plt.plot(sll, label = "Smooth Loss")
plt.legend(loc='best')
plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Smooth Loss', fontsize=16)
plt.show()

















