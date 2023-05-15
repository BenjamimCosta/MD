import numpy as np

class NaiveBayes:
    # Inicializa as variáveis de classe
    def __init__(self):
        self.classes = None
        self.priors = None
        self.means = None
        self.stds = None

    # Calcula as probabilidades, médias e desvios padrão anteriores para cada classe
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = np.zeros(len(self.classes))
        self.means = np.zeros((len(self.classes), X.shape[1]))
        self.stds = np.zeros((len(self.classes), X.shape[1]))

        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[i] = X_c.shape[0] / float(X.shape[0])
            self.means[i, :] = X_c.mean(axis=0)
            self.stds[i, :] = X_c.std(axis=0)
    # Calcula as probabilidades posteriores para cada classe e retorna a classe com a maior probabilidade       
    def predict(self, X):
      n_samples = X.shape[0]
      n_classes = len(self.classes)
      probs = np.zeros((n_samples, n_classes))
      
      for i in range(n_classes):
          prior = self.priors[i]
          likelihood = 1 / np.sqrt(2 * np.pi * self.stds[i]**2) * np.exp(-(X - self.means[i])**2 / (2 * self.stds[i]**2))
          total_likelihood = np.prod(likelihood, axis=1)
          posterior = prior * total_likelihood
          probs[:, i] = posterior
      
      y_pred = self.classes[np.argmax(probs, axis=1)]
      return y_pred

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

n_tests = 10
accuracy = np.zeros(n_tests)

for i in range(n_tests):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    accuracy[i] = accuracy_score(y_test, y_pred)

print('Average accuracy:', np.mean(accuracy))