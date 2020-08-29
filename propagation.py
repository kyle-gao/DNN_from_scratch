import numpy as np
from helper_functions import *


def forward(A_prev,W,b,activation='relu'):
    Z = np.dot(A_prev, W) + b.T
    if activation == 'relu':
        A_next = relu(Z)
    elif activation == 'sigmoid':
        A_next = sigmoid(Z)
    elif activation == 'linear':
        A_next = Z
    elif activation == 'softmax':
        A_next = softmax(Z)
    else:
        A_next = None
        print('Activation not recognized')

    return A_next, Z


def backward(dA,W,A_prev,Z,activation="relu"):

    m = np.shape(Z)[0]

    if activation == 'relu':
        dZ = dA * drelu(Z)
    elif activation == 'sigmoid':
        dZ = dA * dsigmoid(Z)
    elif activation == 'linear':
        dZ = dA
    elif activation == 'softmax':  # Can absorb the derivative of the softmax into the derivative of the loss function
        dZ = np.einsum("ijk,ij->ik", dsoftmax(Z), dA)
    else:
        A_next = None
        print('back Activation not recognized')

    dA_prev = np.dot(dZ, W.T)
    dW = np.dot(A_prev.T, dZ) / m
    dB = np.mean(dZ, axis=0)
    dB = np.reshape(dB, (len(dB), 1))
    return dA_prev, dW, dB


def forward_prop(Ws, Bs, X, middle_activations='relu', last_activation='softmax'):
    As = []
    Zs = [None]
    As.append(X)
    L = len(Ws)

    for i in range(L - 1):
        Anext, Znext = forward(As[-1], Ws[i], Bs[i], middle_activations)
        As.append(Anext)
        Zs.append(Znext)
    Anext, Znext = forward(As[-1], Ws[L - 1], Bs[L - 1], last_activation)
    As.append(Anext)
    Zs.append(Znext)
    return As, Zs


def back_prop(Ws, As, Y, Zs, middle_activations='relu', last_activation='softmax'):
    dWs = []
    dBs = []

    L = len(Ws)

    # The softmax derivative dL/dWij=dL/dai dai/dWij = (Y-A)*dZ which has the same form as the sigmoid derivative
    dA = As[-1] - Y
    if last_activation == 'softmax' or last_activation =='sigmoid':
        last_activation = 'linear'
    dA, dW, dB = backward(dA, Ws[L - 1], As[L - 1], Zs[L], last_activation)
    # The linear backward layer takes into account both softmax and sigmoid final layers.

    dWs.append(dW)
    dBs.append(dB)
    for i in range(L - 2, -1, -1):
        dA_prev, dW, dB = backward(dA, Ws[i], As[i], Zs[i + 1], middle_activations)
        dWs.append(dW)
        dBs.append(dB)
        dA = dA_prev
    dWs.reverse()
    dBs.reverse()
    return dWs, dBs