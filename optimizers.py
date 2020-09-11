"""
Copyright 2020 Yi Lin(Kyle) Gao
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License."""


import numpy as np

def sgd(Ws, Bs, dWs, dBs, vdw=0, vdb=0, sdw=0, sdb=0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-6, t=1):
    """
    Updates the weights and biases according to stocastic gradient descent

    :param Ws: array of weights
    :param Bs: array of biases
    :param dWs: array of derivative weights
    :param dBs: array of derivative biases
    :param vdw: unused
    :param vdb: unused
    :param sdw: unused
    :param sdb: unused
    :param lr: float learning rate
    :param beta1: unused
    :param beta2: unused
    :param eps: unused
    :param t: unused
    :return: Ws, Bs -> updated array of weights and biases
    """
    L = len(Ws)

    Bs = [Bs[i]-lr*dBs[i] for i in range(L)]
    Ws = [Ws[i]-lr*dWs[i] for i in range(L)]

    return Ws, Bs, (vdw, vdb, sdw, sdb)


def momentum_sgd(Ws, Bs, dWs, dBs, vdw=0, vdb=0, sdw=0, sdb=0, lr=0.001, beta1=0.9, beta2= 0.999, eps=1e-6, t=1):
    """
    Updates the weights and biases according to the sgd with momentum

    :param Ws: array of weights
    :param Bs: array of biases
    :param dWs: array of derivative weights
    :param dBs: array of derivative biases
    :param vdw: array of weight momentum
    :param vdb: array of bias momentum
    :param sdw: unused
    :param sdb: unused
    :param lr: float learning rate
    :param beta1: momentum moving average parameter
    :param beta2: unused
    :param eps: unused
    :param t: time step
    :return: Ws, Bs -> updated array of weights and biases
    """
    L = len(Ws)

    vdw = [beta1 * vdw[i] + (1 - beta1) * dWs[i] for i in range(L)]
    vdb = [beta1 * vdb[i] + (1 - beta1) * dBs[i] for i in range(L)]
    # bias correction
    vdw = [vdwi / (1 - np.power(beta1, t)) for vdwi in vdw]
    vdb = [vdbi / (1 - np.power(beta1, t)) for vdbi in vdb]

    Bs = [Bs[i] - lr * vdb[i] for i in range(L)]
    Ws = [Ws[i] - lr * vdw[i] for i in range(L)]

    return Ws, Bs, (vdw, vdb, sdw, sdb)


def rmsprop(Ws, Bs, dWs, dBs, vdw=0, vdb=0, sdw=0, sdb=0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-6, t=1):
    """
    Updates the weights and biases according to the rmsprop

    :param Ws: array of weights
    :param Bs: array of biases
    :param dWs: array of derivative weights
    :param dBs: array of derivative biases
    :param vdw: unused
    :param vdb: unused
    :param sdw: array of squared weight momentum
    :param sdb: array of squared bias momentum
    :param lr: float learning rate
    :param beta1: unused
    :param beta2: squared momentum moving average parameter
    :param eps: stability parameter
    :param t: time step
    :return: Ws, Bs -> updated array of weights and biases
    """
    L = len(Ws)

    sdw = [beta2 * sdw[i] + (1 - beta2) * dWs[i] ** 2 for i in range(L)]
    sdb = [beta2 * sdb[i] + (1 - beta2) * dBs[i] ** 2 for i in range(L)]
    # bias correction
    sdw = [sdwi / (1 - np.power(beta2, t)) + eps for sdwi in sdw]
    sdb = [sdbi / (1 - np.power(beta2, t)) + eps for sdbi in sdb]

    Ws = [Ws[i] - lr * dWs[i] / np.sqrt(sdw[i]) for i in range(L)]
    Bs = [Bs[i] - lr * dBs[i] / np.sqrt(sdb[i]) for i in range(L)]

    return Ws, Bs, (vdw, vdb, sdw, sdb)


def adam(Ws, Bs, dWs, dBs, vdw=0, vdb=0, sdw=0, sdb=0, lr=0.001, beta1=0.9, beta2= 0.999, eps=1e-7, t=1):
    """
    Updates the weights and biases according to the adam optimizer

    :param Ws: array of weights
    :param Bs: array of biases
    :param dWs: array of derivative weights
    :param dBs: array of derivative biases
    :param vdw: array of weight momentum
    :param vdb: array of bias momentum
    :param sdw: array of squared weight momentum
    :param sdb: array of squared bias momentum
    :param lr: float learning rate
    :param beta1: momentum moving average parameter
    :param beta2: squared momentum moving average parameter
    :param eps: stability parameter
    :param t: time step
    :return: Ws, Bs -> updated array of weights and biases
    """
    L = len(Ws)
    lr =lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
    vdw = [beta1 * vdw[i] + (1 - beta1) * dWs[i] for i in range(L)]
    vdb = [beta1 * vdb[i] + (1 - beta1) * dBs[i] for i in range(L)]
    sdw = [beta2 * sdw[i] + (1 - beta2) * np.square(dWs[i]) for i in range(L)]
    sdb = [beta2 * sdb[i] + (1 - beta2) * np.square(dBs[i]) for i in range(L)]

    #bias correction
    vdw = [vdwi / (1 - np.power(beta1, t)) for vdwi in vdw]
    vdb = [vdbi / (1 - np.power(beta1, t)) for vdbi in vdb]
    sdw = [sdwi / (1 - np.power(beta2, t)) for sdwi in sdw]
    sdb = [sdbi / (1 - np.power(beta2, t)) for sdbi in sdb]

    Ws = [Ws[i] - lr * vdw[i] / (np.sqrt(sdw[i])+eps) for i in range(L)]
    Bs = [Bs[i] - lr * vdb[i] / (np.sqrt(sdb[i])+eps) for i in range(L)]

    return Ws, Bs, (vdw, vdb, sdw, sdb)
