import numpy as np
import sys
sys.path.append('../Aula1')
from dataset import Dataset

class VarianceThreshold:
  '''
  Classe que faz seleção/filtração de features de um Dataset utilizando um limite de variância
  Todas as features cuja variância seja menor ou igual a este limite são eliminadas do Dataset

  Argumentos:
      - limite : float -> O valor limite usado para filtrar as features
  Parâmetros estimados:
      - variancia : array de tamanho n (para n features) -> Guarda variância de cada feature do Dataset
  '''
  # Inicializar o transformador com o argumento limite
  def __init__(self, limite):
    self.limite = limite
    self.variancia = None

  # Calcular a variância de cada feature na matriz de entrada
  def fit(self, X):
    self.variancia = np.var(X, axis=0)
    return self

  # Selecionar as features com variância superior ao limite
  def transform(self, X):
    return  X[:, self.variancia > self.limite]

  def fit_transform(self, X):
    return self.fit(X).transform(X)

dataset = Dataset(X=np.array([[0, 2, 0, 3],
                              [0, 1, 4, 3],
                              [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features_names=["f1", "f2", "f3", "f4"],
                      label="y")

selector = VarianceThreshold(0.0)
print("features names:")
print(dataset.features_names)
print("X:")
print(dataset.X)
print("Variância: " + str(selector.variancia))
print("Limite: " + str(selector.limite))
print("----------- fit --------------")
selector = selector.fit(dataset.X)
print("features names:")
print(dataset.features_names)
print("X:")
print(dataset.X)
print("Variância: " + str(selector.variancia))
print("----------- transform --------------")
dataset.X = selector.transform(dataset.X)
print("features names:")
print(dataset.features_names)
print("X:")
print(dataset.X)
