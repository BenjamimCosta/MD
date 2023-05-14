import numpy as np
from dataset import Dataset

class MLP:
    def __init__(self, dataset, hidden_nodes=2, normalize=False):
        self.X, self.y = dataset.getXy()
        self.X = np.c_[np.ones([self.X.shape[0]]), self.X]

        self.h = hidden_nodes
        self.W1 = np.random.randn(hidden_nodes, self.X.shape[1])
        self.W2 = np.random.randn(1, hidden_nodes + 1)

        if normalize:
            self.normalize()
        else:
            self.normalized = False

    def setWeights(self, w1, w2):
        self.W1 = w1
        self.W2 = w2

    def predict(self, instance):
        x = np.r_[1, instance[:self.X.shape[1] - 1]]

        if self.normalized:
            x[1:] = (x[1:] - self.mu) / (self.sigma + 1e-8)

        z2 = self.W1 @ x
        a2 = np.r_[1, sigmoid(z2)]
        z3 = self.W2 @ a2

        return sigmoid(z3)

    def costFunction(self, weights=None):
        m= self.X.shape[0]
        Z2 = self.X @ self.W1.T
        A2 = np.c_[np.ones([Z2.shape[0]]), sigmoid(Z2)]
        Z3= A2 @ self.W2.T
        predictions = sigmoid(Z3)
        sqe = (predictions - self.y.reshape(m,1))**2
        res= np.mean(sqe)/2
        return res

    def build_model(self):
        alpha = 0.01 # learning rate
        epochs = 1000 # number of iterations
        m= self.X.shape[0] # number of examples

        for i in range(epochs):
            # Forward propagation
            Z2 = self.X @ self.W1.T
            A2 = np.c_[np.ones([Z2.shape[0]]), sigmoid(Z2)]
            Z3= A2 @ self.W2.T
            predictions = sigmoid(Z3)

            # Backpropagation
            delta3 = predictions - self.y.reshape(m,1)
            delta2 = delta3 @ self.W2[:, 1:] * sigmoid(Z2) * (1 - sigmoid(Z2))
            gradW2 = delta3.T @ A2 / m
            gradW1 = delta2.T @ self.X / m

            # Gradient descent update
            self.W2 -= alpha * gradW2
            self.W1 -= alpha * gradW1

    def normalize(self):
        self.mu = np.mean(self.X[:, 1:], axis=0)
        self.X[:, 1:] -= self.mu
        self.sigma = np.std(self.X[:, 1:], axis=0)
        self.X[:, 1:] /= (self.sigma + 1e-8)
        self.normalized = True

def sigmoid(x): return 1 / (1 + np.exp(-x))

#Adicionar testes