#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def softmax(x, axis=0):
    y = x - np.max(x, axis=axis)
    z = np.exp(y)
    return z / z.sum(axis=axis, keepdims=True)


def dsoftmax(X):
    derivs = []
    for i in range(X.shape[1]):
        x = X[:,i]
        I = np.eye(x.shape[0])
        derivs.append(softmax(x) * (I - softmax(x).T))
    return np.array(derivs)


def ReLU(x_i):
    """
    x_i (n_features): np array for a single training sample
    """
    return x_i * (x_i > 0)


def dReLU(x_i):
    return 1 * (x_i > 0)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def __init__(self, n_classes, n_features):
        super().__init__(n_classes, n_features)

    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Make prediction 
        y_pred = np.argmax(np.dot(self.W, x_i))

        # update upon mismatch
        if y_i != y_pred:
            self.W[y_i] += x_i
            self.W[y_pred] -= x_i
            

class LogisticRegression(LinearModel):
    def __init__(self, n_classes, n_features):
        super().__init__(n_classes, n_features)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _one_hot(self, y_i):
        return np.eye(self.W.shape[0])[y_i]

    def loss(x_i, y_i, h):
        return (-y_i * np.log(h) - (1 - y_i) * np.log(1 - h)).mean()

    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b 
        y_one_hot = self._one_hot(y_i)

        z = np.dot(self.W, x_i)
        h = self._sigmoid(z)
        error = y_one_hot - h

        grad = np.outer(error, x_i)
        self.W += learning_rate * grad


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, n_layers):
        # Constants
        self.n_classes = n_classes
        self.n_layers = 3

        # Iterables
        self.W = np.array([.1 * np.ones((hidden_size, n_features)), 
                           .1 * np.ones((hidden_size, hidden_size)),
                           .1 * np.ones((n_classes, hidden_size))])
        self.biases = np.array([.1 * np.ones(hidden_size),
                                .1 * np.ones(hidden_size), 
                                .1 * np.ones(n_classes)])          
        self.activate = np.array([ReLU, ReLU, softmax])

    def _one_hot(self, y):
        return np.eye(self.n_classes)[y].T

    def _forward(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        scores = []
        score = X.T
        for i in range(self.n_layers):
            score = np.dot(self.W[i], score) + np.tile(self.biases[i], (score.shape[1], 1)).T
            score = self.activate[i](score)
            scores.append(score)
        return scores[-1], scores[:-1]

    def predict(self, X):
        score = X.T
        for i in range(self.n_layers):
            score = np.dot(self.W[i], score) + np.tile(self.biases[i], (score.shape[1], 1)).T
            score = self.activate[i](score)
        return np.argmax(score, axis=0)
        
    def _backprop(self, X, y_pred, y, hiddens):
        grad_weights = []
        grad_biases = []

        grad_z = y_pred - y
        for i in range(self.n_layers - 1, -1, -1):
            h = X.T if i == 0 else hiddens[i - 1]
            grad_weights.append(grad_z.dot(h.T))
            grad_biases.append(grad_z.sum(axis=1, keepdims=False))

            grad_h = self.W[i].T.dot(grad_z)
            grad_z = grad_h * dReLU(h)

        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        y = self._one_hot(y)
        y_pred, hiddens = self._forward(X)
        grads, grad_biases = self._backprop(X, y_pred, y, hiddens)

        for i in range(len((self.W))):
            self.W[i] -= learning_rate * grads[i]
            self.biases[i] -= learning_rate * grad_biases[i]


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.title('Accuracy for the logistic regression model per epoch')
    plt.savefig('Logistic_acc.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()


def backpropagation(self,y, z_s, a_s):
  dw = []  # dC/dW
  db = []  # dC/dB
  deltas = [None] * len(self.weights)  # delta = dC/dZ  known as error for each layer
  # insert the last layer error
  deltas[-1] = ((y-a_s[-1])*(self.getDerivitiveActivationFunction(self.activations[-1]))(z_s[-1]))
  # Perform BackPropagation
  for i in reversed(range(len(deltas)-1)):
    deltas[i] = self.weights[i+1].T.dot(deltas[i+1])*(self.getDerivitiveActivationFunction(self.activations[i])(z_s[i]))        
    batch_size = y.shape[1]
    db = [d.dot(np.ones((batch_size,1)))/float(batch_size) for d in deltas]
    dw = [d.dot(a_s[i].T)/float(batch_size) for i,d in enumerate(deltas)]
    # return the derivitives respect to weight matrix and biases
    return dw, db