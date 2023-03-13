import numpy as np
import sys
sys.path.insert(1, './Aula1')
from dataset import Dataset
class VarianceThreshold:
    '''
    Classe que faz seleção/filtração de features de um Dataset utilizando um limite de variância
    Todas as features cuja variância seja menor ou igual a este limite são eliminadas do Dataset

    Argumentos:
        - threshold : float -> O valor limite usado para filtrar as features
    Parâmetros estimados:
        - variance : array de tamanho n (para n features) -> Guarda variância de cada feature do Dataset
    '''

    def __init__(self, threshold: float = 0.0):
        if threshold < 0:
            raise ValueError("Threshold deve ser não negativo")

        #Argumentos
        self.threshold = threshold

        #Parâemtros estimados
        self.variance = None

    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        #calcula variância de cada uma das features (por coluna)
        self.variance = np.var(dataset.X, axis=0)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        X = dataset.X

        #self.variance é um array com a variância de cada coluna, logo features_name_mask é um array que em cada posição tem um booleano que informa se aquela coluna 
        #satisfaz a condição ou não
        features_names_mask = self.variance > self.threshold
        #Remove colunas que não satisfazem a condição, ou seja, que têm variância maior que o limite
        X = X[:, features_names_mask]
        #Remove features_names com este mesmo array de booleanos, tirando as mesmas colunas que foram removidas na linha acima
        features_names = np.array(dataset.features_names)[features_names_mask]
        #Retorna Dataset com as features filtradas de acordo com o limite de variância
        return Dataset(X=X, y=dataset.y, features_names=list(features_names), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        #Dá fit dos dados (calcular vetor de variâncias) e depois transforma-os (filtra as features do dataset fornecido)
        self.fit(dataset)
        return self.transform(dataset)


if __name__ == '__main__':
    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]]),
                      y=np.array([0, 1, 0]),
                      features_names=["f1", "f2", "f3", "f4"],
                      label="y")

    selector = VarianceThreshold()
    print("features names:")
    print(dataset.features_names)
    print("X:")
    print(dataset.X)
    print("Variância: " + str(selector.variance))
    print("Limite: " + str(selector.threshold))
    print("----------- fit --------------")
    selector = selector.fit(dataset)
    print("features names:")
    print(dataset.features_names)
    print("X:")
    print(dataset.X)
    print("Variância: " + str(selector.variance))
    print("----------- transform --------------")
    dataset = selector.transform(dataset)
    print("features names:")
    print(dataset.features_names)
    print("X:")
    print(dataset.X)

