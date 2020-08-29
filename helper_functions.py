import numpy as np


def sigmoid(Z):
    """sigmoid function"""
    return  1/(1+np.exp(-Z))


def dsigmoid(Z):
    """sigmoid derivative"""
    return sigmoid(Z)*(1-sigmoid(Z))


def relu(Z):
    """returns the element-wise relu of Z, fast implementation
    Z -- an array of floats"""
    return Z * (Z > 0)


def drelu(Z):
    #relu derivative
    return 1. * (Z > 0)


def softmax(Z,eps=1e-10):
    """Numerically stable softmax
    Given (m,n) input, returns softmax over the last dimension"""
    shiftZ = Z - np.max(Z,axis=-1,keepdims=True)
    expZ=np.exp(shiftZ)
    total=np.sum(expZ,axis=-1,keepdims=True)+eps
    return expZ/total


def dsoftmax(Z):
    """Given a (m,n) matrix, returns a (m,n,n) jacobian matrix"""
    m,n=np.shape(Z)
    softZ=(softmax(Z))
    prodtensor=np.einsum("ij,ik->ijk",softZ,softZ)
    diagtensor=np.einsum('ij,jk->ijk', softZ, np.eye(n, n))
    return diagtensor-prodtensor


def one_hot(Y, n_class):
    """
    Returns the one_hot representation (m,n_class) as a numpy array

    Y - (m,) array of classes from (0 to n_class)
    n_class - an integer representing the number of classes
    """
    m = np.shape(Y)[0]

    O_h = np.zeros((m, n_class))
    O_h[range(m), Y] = 1

    return O_h


def initialize_parameters(layers):
    """ 
    Input:
    layers -- list of int:layer sizes, note: layer[0] is input layer, layer[L] is output layer

    Output:
    Ws,Bs -- lists of np arrays, the randomly initialized weights and zero initialized biases"""""

    L = len(layers)
    Ws = []
    Bs = []

    for l in range(1, L):
        wl = np.random.randn(layers[l - 1], layers[l])
        bl = np.zeros((layers[l], 1))
        Ws.append(wl)
        Bs.append(bl)
    return Ws, Bs


def get_rand_minibatch(X, Y, minibatch_size=64):
    """
    Returns shuffled minibatches given X, Y
    :param X: an array of features (m, features)
    :param Y: an array of targets (m,)
    :param minibatch_size: int
    :return: (mini_X, mini_Y) each of batch size minibatch_size
    """

    m = X.shape[0]
    minibatches = []

    rand_perm = np.random.permutation(m)
    shuffled_X = X[rand_perm, :]
    shuffled_Y = Y[rand_perm]
    num_batches = int(np.floor(m / minibatch_size))

    for b in range(num_batches):
        mini_X = shuffled_X[b * minibatch_size:(b + 1) * minibatch_size, :]
        mini_Y = shuffled_Y[b * minibatch_size:(b + 1) * minibatch_size]
        minibatches.append((mini_X, mini_Y))
    if m % minibatch_size != 0:  # left over minibatch
        mini_X = shuffled_X[num_batches * minibatch_size:, :]
        mini_Y = shuffled_Y[num_batches * minibatch_size:]
        minibatches.append((mini_X, mini_Y))

    return minibatches
