#!/usr/bin/env python

"""
Deep Learning Homework 1
Author: Willem van der Spek
Co-Author: Nuno Damos
"""


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
    """
    Numerically stable softmax activation function.
    """
    y = x - np.max(x, axis=axis) # Wrap around largest 
    z = np.exp(y)
    return z / z.sum(axis=axis, keepdims=True)


def ReLU(X):
    return X * (X > 0)


def dReLU(X):
    return 1 * (X > 0)


def leaky_ReLU(X, alpha=0.01):
    return np.where(X > 0, X, X * alpha)


def dleaky_ReLU(X, alpha=0.01):
    return np.where(X > 0, 1, alpha)


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
        args:
            x_i (n_features): a single training example
            y_i (scalar): the gold label for that example
        other arguments are ignored

        Update the weights for the Perceptron model for a single training epoch.
        """
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

    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        Args:
            x_i (n_features): a single training example
            y_i: the gold label for that example
            learning_rate (float): keep it at the default value for your plots
        Update the weights for the logistic regression model for a single epoch.

        """
        # Q1.1b 
        y_one_hot = self._one_hot(y_i) 
        z = np.dot(self.W, x_i)
        h = self._sigmoid(z)
        error = y_one_hot - h

        # Update
        grad = np.outer(error, x_i)
        self.W += learning_rate * grad


class MLP(object):
    """
    Q1.2b. This MLP skeleton code allows the MLP to be used in place of the
    linear models with no changes to the training loop or evaluation code
    in main().
    
    Attributes:
        n_classes: Number of classes to predict
        W: Model weights 
        b: Model biases 
        activate: Activation funcs used at each layer.
    """
    def __init__(self, n_classes, n_features, hidden_size, n_layers):
        """
        Args: 
            n_classes: Number of classes to predict
            n_features: Number of features 
            hidden_size: Number of hidden nodes to be used

        Inits MLP with normal random weights and zero biases.
        """
        self.n_classes = n_classes
        self.n_layers = n_layers + 1

        # Metaparameters for initialising weights.
        mu = .1
        sigma = np.sqrt(0.1)

        self.W = []
        self.b = []
        self.activate = []

        for i in range(n_layers):
            size = (hidden_size, n_features) if not i else (hidden_size, hidden_size)
            self.W.append(np.random.normal(mu, sigma, size=size)), 
            self.b.append(np.zeros((hidden_size, 1)))
            self.activate.append(leaky_ReLU)
        self.W.append(np.random.normal(mu, sigma, size=(n_classes, hidden_size)))
        self.b.append(np.zeros((n_classes, 1)))
        self.activate.append(softmax)

    def _one_hot(self, y):
        """
        Args:
            y <int>: gold label as integer

        Returns the one-hot encoded representation of an integer:
        Example for n_classes = 3: y = 1 => np.array([0, 1, 0]).T
        """
        return np.eye(self.n_classes)[y].T

    def _forward(self, X):
        """
        Args:
            X (n_samples, n_features): The initial training data.
        Returns:
            scores: Output for all layers.

        Compute the forward pass of the network at training time. 
        Saves the output values of hidden nodes and output nodes. 
        """
        scores = []
        score = X.T
        for i in range(self.n_layers):
            score = np.dot(self.W[i], score) + self.b[i]
            score = self.activate[i](score)
            scores.append(score)
        return scores[-1], scores[:-1]
        
    def _backprop(self, X, y_hat, y, hiddens):
        """
        Args:
            X (n_samples, n_features): The initial training data.
            y_hat (n_samples): Probabilistic vector for the class predictions.
            y (n_samples): The gold labels per sample
            hiddens: List of layer outputs at training time.
        Returns:
            grad_weights: List of gradients for the weights.
            grad_biases: List of gradients for the biases.

        Compute the forward pass of the network at inference time.
        Returns the labels through their integer representations. 
        """
        grad_weights = []
        grad_biases = []

        grad_z = y_hat - y  
        for i in range(self.n_layers - 1, -1, -1):
            grad_biases.append(grad_z.sum(axis=1, keepdims=True))
            if i == 0:
                grad_weights.append(grad_z.dot(X))
            else:
                h = hiddens[i - 1]
                grad_weights.append(grad_z.dot(h.T))

                # Get gradient at previous layer
                grad_h = self.W[i].T.dot(grad_z)
                grad_z = grad_h * dleaky_ReLU(h)

        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases

    def predict(self, X):
        """
        Args:
            X (n_samples, n_features): The initial training data.
        Returns:
            y_hat List[int]: Prediction given as list of integers

        Compute the forward pass of the network at inference time.
        Returns the labels through their integer representations. 
        """
        score = X.T
        for i in range(self.n_layers):
            score = np.dot(self.W[i], score) + self.b[i]
            score = self.activate[i](score)
        y_hat = np.argmax(score, axis=0) # Get most likely label as int.
        return y_hat

    def evaluate(self, X, y):
        """
        Args:
            X (n_examples x n_features)
            y (n_examples): gold labels

        
        Identical to LinearModel.evaluate()
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Args:
            X (n_samples, n_features): The initial training data.
            y (n_samples): The gold labels per sample
            learning_rate

        Train the model for a single epoch. Updates the model parameters through
        forward and backward propagation. 
        """
        y = self._one_hot(y)
        y_hat, hiddens = self._forward(X)
        grads, grad_biases = self._backprop(X, y_hat, y, hiddens)

        # Update model parameters
        for i in range(len(self.W)):
            self.W[i] -= learning_rate * grads[i]
            self.b[i] -= learning_rate * grad_biases[i]


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs[1::len(epochs) // 10], rotation=90)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.title('Accuracy for the MLP model per epoch')
    plt.savefig('MLP_acc.png')


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
