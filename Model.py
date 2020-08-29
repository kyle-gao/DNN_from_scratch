import numpy as np
import matplotlib.pyplot as plt
import mnist
import copy
from helper_functions import *
from propagation import *
from loss_functions import *
import optimizers

class Model():

    def __init__(self, layers, middle_activations = 'relu', last_activation='softmax'):

        self.layers = layers
        self.middle_activations = middle_activations
        self.last_activation = last_activation
        self.L = len(layers)
        self.costs=[]

        self.Ws, self.Bs = initialize_parameters(layers)

    def optimize(self, X, Y, optimizer, lr = 0.001, epochs = 20, batch_size = 128):
        """
        Optimizes the weights self.Ws and biases self.Bs of the Model
        :param X: A flatten array of input features (m, features)
        :param Y: A one hot encoded array of labels (m, n_classes)
        :param optimizer: An optimizers function from the optimizers.py module
        :param epochs: Number training epochs
        :param lr: learning rate
        :param batch_size: mini batch size
        """
        minibatches = get_rand_minibatch(X, Y, batch_size)
        t = 1
        m = np.shape(X)[0]
        vdw=[np.zeros(np.shape(i)) for i in self.Ws]
        vdb=[np.zeros(np.shape(i)) for i in self.Bs]
        sdw=[np.zeros(np.shape(i)) for i in self.Ws]
        sdb=[np.zeros(np.shape(i)) for i in self.Bs]

        for i in range(epochs):

            cost_total = 0

            for minibatch in minibatches:

                (mini_X, mini_Y) = minibatch

                # Forward propagation
                As, Zs = forward_prop(self.Ws, self.Bs, mini_X, middle_activations=self.middle_activations, last_activation=self.last_activation)
                Yhat = As[-1]
                cost_total += categorical_crossentropy(mini_Y, Yhat)

                dWs, dBs = back_prop(self.Ws, As, mini_Y, Zs, middle_activations=self.middle_activations, last_activation=self.last_activation)

                self.Ws, self.Bs,_ = optimizer(self.Ws, self.Bs, dWs, dBs, vdw, vdb, sdw, sdb, lr=lr, beta1=0.9, beta2=0.999, eps=1e-6, t=t)

                t = t + 1

            cost = cost_total / m
            print("loss:", cost)
            self.costs.append(cost)

    def predict(self, X):
        """
        :param X: an input array (m,features)
        :return:  the model's prediction (m, prediction)
        """
        As, Zs = forward_prop(self.Ws, self.Bs, X)
        Yhat = As[-1]
        prediction = np.argmax(Yhat, axis=-1)
        return prediction

    def evaluate(self,X, Y):
        """
        :param X: an input array (m,features)
        :param Y: a target array (m,)
        :return: the prediction accuracy
        """
        predicts = self.predict(X)
        acc = np.count_nonzero(predicts == Y) / len(Y)
        return acc


def main():
    train_X = mnist.train_images() / 255.0
    train_Y = mnist.train_labels()

    test_X = mnist.test_images() / 255.0
    test_Y = mnist.test_labels()

    train_X = np.reshape(train_X, (-1, 28 * 28))
    test_X = np.reshape(test_X, (-1, 28 * 28))

    train_Y_OH = one_hot(train_Y,10)
    model = Model(layers=[784,28,10], middle_activations = 'relu', last_activation='softmax')
    model.optimize(train_X, train_Y_OH, optimizers.adam, epochs = 10, batch_size = 256, lr = 0.01)
    print("accuracy = ", model.evaluate(train_X,train_Y))


if __name__ == "__main__":
    main()

