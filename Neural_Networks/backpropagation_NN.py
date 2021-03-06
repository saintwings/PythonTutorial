import numpy as np
import math
def sigmoid(x):      # sigmoid function
    return 1 /(1+(math.e**-x))

def deriv_sigmoid(y): #the derivative of the sigmoid function
    return y * (1.0 - y)

alpha=.1    #this is the learning rate
X = np.array([  [.35,.21,.33],
    [.2,.4,.3],
    [.4,.34,.5],
    [.18,.21,16] ])
y = np.array([[0],
        [1],
        [1],
        [0]])
np.random.seed(1)
#We randomly initialize the layers
theta0 = 2*np.random.random((3,4)) - 1
theta1 = 2*np.random.random((4,1)) - 1

for iter in range(205000): #here we specify the amount of training rounds.
    # Feedforward the input like we did in the previous exercise
    input_layer = X
    l1 = sigmoid(np.dot(input_layer,theta0))
    l2 = sigmoid(np.dot(l1,theta1))

    # Calculate error
    l2_error = y - l2

    if (iter% 1000) == 0:
        print "Neuralnet accuracy:" + str(np.mean(1-(np.abs(l2_error))))

    # Calculate the gradients in vectorized form
    # Softmax and bias units are left out for instructional simplicity
    l2_delta = alpha*(l2_error*deriv_sigmoid(l2))
    l1_error = l2_delta.dot(theta1.T)
    l1_delta = alpha*(l1_error * deriv_sigmoid(l1))

    theta1 += l1.T.dot(l2_delta)
    theta0 += input_layer.T.dot(l1_delta)