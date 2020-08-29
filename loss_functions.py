import numpy as np
def categorical_crossentropy(Y, Yhat, eps=1e-8):
    """
    Computes the categorical cross-entropy for one hot encoded labels
    Y - (m,dim_y) array of labels m=examples
    Yhat - (m,dim_y) array of logits
    """
    m = np.shape(Y)[0]

    Yhat = np.clip(Yhat, eps, 1 - eps)  # for numerical stability if this is used, the output of the last layer must be normalized

    J = -np.sum(Y * np.log(Yhat), axis=-1)

    return np.sum(J, axis=0) / (m)