
import numpy as np
import random
import torch
from os.path import exists
import torch.nn as nn
import torch.nn.functional as F
import sklearn.base
import math

from copy import deepcopy



class PyTorchMLP():
    def __init__(self, path = None, verbose=False, output_layer_activation=None, hidden_layer_sizes=[25], hidden_layer_activation=[], max_iter=5000, tol=0.0001, momentum=0.9, learning_rate=1e-6):
        self._model = None
        self._dtype = torch.float
        self._device = torch.device("cpu")
        self._y_shape = None
        self._x_shape = None
        self._max_iter = max_iter
        self._tol = tol
        self._momentum = momentum
        self._learning_rate = learning_rate
        self._hidden_layer_sizes = hidden_layer_sizes
        self._hidden_layer_activation = hidden_layer_activation
        self._output_layer_activation = output_layer_activation
        self._history = None
        if exists(path):
            self._path = path
            self.create_model()
        else:
            self._path = None
        self._verbose = verbose
        self._clear_history()
        self._lock_save = False

    def save(self, path):
        if self._lock_save == False:
            print("Saving model")
            torch.save(self._model, path+ "/" + str(self) + ".pytorch")

    def load(self, path):
        self._model = torch.load(path)
        self._model.eval()

    def create_model(self, in_features=None, out_features=None):
        if self._path is not None:
            print("Loading model from the path: ", self._path)
            self.load(self._path)
            print(self._model)
        else:

            self._model = torch.nn.Sequential()

            layers = [in_features] + self._hidden_layer_sizes + [out_features]

            for idx, dim in enumerate(layers):
                if (idx < len(layers) - 1):
                    module = torch.nn.Linear(dim, layers[idx + 1])
                    self._model.add_module("linear" + str(idx), module)

                if (idx < len(layers) - 2):
                    if len(self._hidden_layer_activation) > idx:
                        self._model.add_module(self._hidden_layer_activation[idx].__class__.__name__ + str(idx), deepcopy(self._hidden_layer_activation[idx]))
                    else:
                        self._model.add_module("relu" + str(idx), torch.nn.ReLU())

            if self._output_layer_activation is not None:
                self._model.add_module(self._output_layer_activation.__class__.__name__ + str(idx), deepcopy(self._output_layer_activation))

            module = torch.nn.Flatten(0, 1)
            #module = nn.LogSoftmax(dim=1)
            self._model.add_module("Flatten", module)

            print("The following model has been built")
            print(self._model)

        self._criterion = torch.nn.MSELoss(reduction='sum')
        #self._criterion = torch.nn.NLLLoss()
        self._optimizer = torch.optim.SGD(self._model.parameters(), lr=self._learning_rate, momentum=self._momentum)
        #self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.003)



    def _train_model(self, X,y):
        X = torch.tensor(X, dtype=self._dtype, device=self._device)
        y = torch.tensor(y, dtype=self._dtype, device=self._device)

        en_early = True
        last_loss = 0
        for t in range(self._max_iter):

            y_pred = self._model(X)
            loss = self._criterion(y_pred, y)
            if self._verbose is True:
                #if (t + 1) % self._max_iter == 0:
                print(t, loss.item())

            z = loss.item()
            if t >= 15:
                #or (en_early and (z - last_loss) > 0)
                if last_loss is not None and (self._tol is not None and abs(z - last_loss) < self._tol):
                    if self._verbose is True:
                        print("Early stopping at iteration t =", t, abs(last_loss - z))
                    break

            last_loss = z
            self._history["loss"].append(z)

            if math.isnan(z):
                self._lock_save = True
                break

            # Zero gradients, perform a backward pass, and update the weights.
            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

        return self

    def fit(self, X,y):
        self._clear_history()
        X = torch.tensor(X, dtype=self._dtype, device=self._device)
        y = torch.tensor(y, dtype=self._dtype, device=self._device)

        #always create a new model
        self._y_shape = 1
        self._x_shape = X.shape[1]
        if len(y.shape) > 1:
            self._y_shape = y.shape[1]
        self.create_model(X.shape[1], self._y_shape)

        return self._train_model(X,y)

    def partial_fit(self, X,y):
        if self._x_shape is None or self._y_shape is None:
            return self.fit(X,y)
        return self._train_model(X, y)

    def predict(self, X):
        if self._model is None:
            self.create_model(X.shape[1], self._y_shape)

        X = torch.tensor(X, dtype=self._dtype, device=self._device)
        y_pred = self._model(X).detach().numpy()

        if  self._y_shape == 1:
            return y_pred.flatten()

        return y_pred

    def __str__(self):
        string = "NN-" + "-".join([str(x) for x in self._hidden_layer_sizes]) + " " + "-".join([x.__class__.__name__ for x in self._hidden_layer_activation])

        if self._output_layer_activation is not None:
            string += "out: " + self._output_layer_activation.__class__.__name__

        print(string)
        return string

    def _clear_history(self):
        self._history = {
            "loss": []
        }

    def get_history(self, deep=False):
        return self._history

    def get_params(self, deep=False):
        pass

    def set_params(self, **params):
        return