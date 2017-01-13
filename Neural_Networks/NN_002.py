import neurolab as nl
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sknn.mlp import Classifier, Layer

plt.style.use('ggplot')
# Create train samples
x = np.linspace(-10,10, 60)
y = np.cos(x) * 0.9
size = len(x)
x_train = x.reshape(size,1)
y_train = y.reshape(size,1)

# Create network with 4 layers and random initialized
# just experiment with the amount of layers

d=[[1,1],[45,1],[45,45,1],[45,45,45,1]]
for i in range(4):
    net = nl.net.newff([[-10, 10]],d[i])
    print net
    train_net=nl.train.train_gd(net, x_train, y_train, epochs=1000, show=100)
    outp=net.sim(x_train)
# Plot results (dual plot with error curve and predicted values)
    import matplotlib.pyplot
    plt.subplot(2, 1, 1)
    plt.plot(train_net)
    plt.xlabel('Epochs')
    plt.ylabel('squared error')
    x2 = np.linspace(-10.0,10.0,150)
    y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
    y3 = outp.reshape(size)
    plt.subplot(2, 1, 2)

    plt.suptitle([i ,'hidden layers'])
    plt.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
    plt.legend(['y predicted', 'y_target'])
    plt.show()