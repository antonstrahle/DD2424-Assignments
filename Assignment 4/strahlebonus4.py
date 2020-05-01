import numpy as np
import pandas as pd
import pickle
import math
import matplotlib.pyplot as plt
import scipy.stats as sc
import json

mainTweets = ["condensed_2018.json", "condensed_2017.json"]

def loadTweets(files, tweetEnder):
	
	tweets = []
	fulltext = ""
	
	for fn in files:
	
		with open("../../TextData/" + fn, 'r', encoding = 'utf8') as f:
			
			data = json.load(f)
			
			for i in range(len(data)):
				tw = data[i]["text"] + tweetEnder
				
				if tw[0] not in ['"', "'"]: #Remove quotes that are not technically from trumpS
					
					if tw[0] + tw[1] != "RT": #Remove retweets
				
						tweets.append(tw) #¤ used as "end of tweet"
						fulltext += tw
	
	chars = list(set(fulltext))
	toChar = sorted(chars, key=str.lower)
	toInd = {}
	
	i = 0
	for char in toChar:
		toInd[char] = i
		i += 1
	
	return tweets, chars, toChar, toInd


class RNN(object):
	 
	def __init__(self, seqLength, eta = 0.1, m = 100, sig = 0.01, 
			  files = mainTweets, 
			  te = "¤"):
		
		self.tweets, self.chars, self.toChar, self.toInd = loadTweets(files, te)
		self.nChars = len(self.chars)
		self.eta = eta
		self.te = te
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
			nc = self.toChar[xind]
			
			if nc == self.te:
				break
			
			else:
				out += nc
				
		print(out + str("\n"))
	
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
		
		return l/len(X), p, o, h, a, x
	
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
	
	def train(self, itr = 50000):
		
		params = {"V": self.V, "U": self.U, "W": self.W, "c": self.c, "b":self.b}
		
		ada = {"W": np.zeros_like(self.W), "U": np.zeros_like(self.U), "V": np.zeros_like(self.V),
			   "b": np.zeros_like(self.b), "c": np.zeros_like(self.c)}
		
		tweetID = 0
		start = 0
		data = self.tweets[tweetID]
		h = np.zeros((self.m, 1))
		
		for i in range(itr):
			data = self.tweets[tweetID]
				
			if (start+1)*self.seqLength > len(self.tweets[tweetID]) - 4: #new tweet if end sequence cant fit remaining tweet
				
				x = [self.toInd[char] for char in data[start*self.seqLength:(len(self.tweets[tweetID])-1)]]
				y = [self.toInd[char] for char in data[(start*self.seqLength + 1):len(self.tweets[tweetID])]]
				
				tweetID += 1
				start = 0
				h = np.zeros((self.m, 1)) #resetting upon entering new tweet
				
				if tweetID > len(self.tweets) - 1: #if out of tweets, star over
					tweetID = 0
					print("Out of tweets \n")
					
			else:
				
				x = [self.toInd[char] for char in data[start*self.seqLength:(start+1)*self.seqLength]]
				y = [self.toInd[char] for char in data[(start*self.seqLength + 1):((start+1)*self.seqLength + 1)]]
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
			
			if i % 5000 == 0:
				h = np.zeros((self.m, 1)) #new tweet <=> clean slate
				print("After " + str(i) + " iterations. Smooth loss: " + str(sl))
				self.synthesize(h, 26, 140) #140 for max tweet length
				print("")
		
		return b, c, U, W, V, h

rnn = RNN(50, m = 100)

rnn.b, rnn.c, rnn.U, rnn.W, rnn.V, h = rnn.train()

print("Printing Fully Trained Tweets: ")

for i in range(len(rnn.tweets[1])):
	print("With regular h")
	rnn.synthesize(h, rnn.toInd[rnn.tweets[1][i]], 140)
	print("With zero h")
	halt = np.zeros((self.m, 1))
	rnn.synthesize(halt, rnn.toInd[rnn.tweets[1][i]], 140)

























