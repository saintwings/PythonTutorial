import numpy as np
import math

def sigmoid(x): # sigmoid function
    return 1 /(1+(math.e**-x))

def deriv_sigmoid(y): #the derivative of the sigmoid function
    return y * (1.0 - y)

alpha=.1    #this is the learning rate

X = np.array([ [.35,.21,.33],
                [.2,.4,.3],
                [.4,.34,.5],
                [.18,.21,16] ])

y = np.array([[0],
                [1],
                [1],
                [0]])

#We randomly initialize the layers
theta0 = 2*np.random.random((3,4)) - 1
theta1 = 2*np.random.random((4,1)) - 1

for iter in range(205000): #here we specify the amount of training rounds.
    # Feedforward the input like we did in the previous exercise
    input_layer = X
    l1 = sigmoid(np.dot(input_layer, theta0))
    l2 = sigmoid(np.dot(l1, theta1))
    # Calculate error
    l2_error = y - l2