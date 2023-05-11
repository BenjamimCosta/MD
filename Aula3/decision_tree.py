import numpy as np
import unittest
from sklearn.metrics import accuracy_score, precision_score

class Node:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index # índice da característica usada para a divisão
        self.threshold = threshold # valor do limite usado para a divisão
        self.value = value # valor do nó (elemento mais frequente ou média) se for folha
        self.left = left # subárvore à esquerda do nó
        self.right = right # subárvore à direita do nó

def _most_common_label(y):
    # Retornar o elemento mais frequente no vetor y
    unique, counts = np.unique(y, return_counts=True)
    # Obter os índices dos elementos com a maior contagem
    max_indices = np.argwhere(counts == counts.max()).flatten()
    # Se houver mais de um elemento com a maior contagem, escolher um aleatoriamente -> (Majority Voting) <-
    if len(max_indices) > 1:
        max_index = np.random.choice(max_indices)
    else:
        max_index = max_indices[0]
    return unique[max_index]

def _entropy(y):
    # Calcular a entropia de um vetor y
    # Quanto maior a entropia, maior a incerteza da previsão, logo o objetivo é reduzir a entropia

    # Obtemos o vetor com os elementos de y sem repetições e o número de vezes que cada um desses elementos aparece em y:
    unique, counts = np.unique(y, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
    return entropy

def _gini_index(y):
    # Calcular o índice de Gini de um vetor y
    # Quanto maior o índice de Gini, maior a impureza do conjunto, logo o objetivo é reduzir o índice de Gini

    # Obtemos o vetor com os elementos de y sem repetições e o número de vezes que cada um desses elementos aparece em y:
    unique, counts = np.unique(y, return_counts=True)

    probabilities = counts / counts.sum()
    gini_index = 1 - sum(probabilities**2)
    return gini_index

def _information_gain(y, X_column, threshold, criterion):
    # Calcular o ganho de informação de um critério de divisão usando a entropia como medida de impureza
    if criterion == 'entropy':
        parent = _entropy(y)
    elif criterion == 'gini':
        parent = _gini_index(y)
    # Dividir o vetor X_column em dois subconjuntos
    left_indices, right_indices = _split(X_column, threshold)

    if len(left_indices) == 0 or len(right_indices) == 0:
        return 0

    # Calcular a entropia ponderada dos subconjuntos
    n = len(y)
    len_left, len_right = len(left_indices), len(right_indices)
    if criterion == 'entropy':
        left, right = _entropy(y[left_indices]), _entropy(y[right_indices])
    elif criterion == 'gini':
        left, right = _gini_index(y[left_indices]), _gini_index(y[right_indices])

    child = (len_left / n) * left + (len_right / n) * right

    # Calcular o ganho de informação
    ig = parent - child
    return ig

def _gain_ratio(y, X_column, threshold):
    # Calcular a razão de ganho de um critério de divisão usando o ganho de informação e a entropia do atributo como medidas
    info_gain = _information_gain(y, X_column, threshold, 'entropy')

    # Dividir o vetor X_column em dois subconjuntos
    left_indices, right_indices = _split(X_column, threshold)

    if len(left_indices) == 0 or len(right_indices) == 0:
        return 0

    # Calcular a entropia do atributo
    n = len(y)
    n_left, n_right = len(left_indices), len(right_indices)
    p_left, p_right = n_left / n, n_right / n
    attr_entropy = -p_left * np.log2(p_left) - p_right * np.log2(p_right)

    # Calcular a razão de ganho
    gain_ratio = info_gain / attr_entropy
    return gain_ratio

def _split(X_column, threshold):
    # Dividir um vetor X_column em dois subconjuntos baseados num limite
    left_indices = np.argwhere(X_column <= threshold).flatten()
    right_indices = np.argwhere(X_column > threshold).flatten()
    return left_indices, right_indices


class DecisionTree:
    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2, max_leaf_nodes=20):
        self.criterion = criterion  # critério de divisão: 'entropy', 'gini' ou 'gain_ratio'
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split # número mínimo de amostras para dividir um nó (para pre-prunning -> Size Cutof)
        self.max_leaf_nodes = max_leaf_nodes # número máximo de leaf nodes na árvore (para pre-prunning -> Maximum Depth Cutof)

    def fit(self, X, y):
        self.tree_ = self._grow_tree(X, y, 0, 0)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _grow_tree(self, X, y, depth, n_leafs):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        if (depth > self.max_depth or n_labels == 1 or n_samples < self.min_samples_split or n_leafs >= self.max_leaf_nodes):
            leaf_value = _most_common_label(y)
            n_leafs += 1 #n_leafs n esta bem
            return Node(value=leaf_value)
        
        feature_indices = np.arange(n_features)
        best_feature, best_threshold = self._best_criteria(X, y, feature_indices)
        left_indices, right_indices = _split(X[:, best_feature], best_threshold)

        left = self._grow_tree(X[left_indices, :], y[left_indices], depth+1, n_leafs) #X[left_indices, :] -> linhas com os left_indices e todas as colunas dessas linhas
        right = self._grow_tree(X[right_indices, :], y[right_indices], depth+1, n_leafs)

        # Atribuir um valor ao nó usando o método _most_common_label
        node_value = _most_common_label(y)
        print(depth)
        return Node(best_feature, best_threshold, node_value, left, right)

    def _predict(self, inputs):
        node = self.tree_
        if node is None:
            return None
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


    def _best_criteria(self, X, y, feature_indices):
        best_gain = -1
        split_index, split_threshold = None, None
        #para cada feature (coluna) vemos qual a melhor feature para dividir a árvore e nessa feature qual o melhor limite a ser usado para a divisão
        #se limite = 0.5, de um lado ficam com a caracteristica menor ou igual a 0.5 e do outro maior que 0.5
        for feature_index in feature_indices:
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                if self.criterion == 'entropy' or self.criterion == 'gini':
                    gain = _information_gain(y, X_column, threshold, self.criterion)
                elif self.criterion == 'gain_ratio':
                    gain = _gain_ratio(y, X_column, threshold)
                else:
                    raise ValueError("Invalid criterion: {}".format(self.criterion))
                if gain > best_gain:
                    best_gain = gain
                    split_index = feature_index
                    split_threshold = threshold

        return split_index, split_threshold


# Classe de teste
class TestDecisionTree(unittest.TestCase):

    # Gerar um conjunto de dados aleatório
    def generate_random_dataset(self, n_samples, n_features, n_classes):
        # Gerando as características aleatoriamente
        X = np.random.rand(n_samples, n_features)
        # Gerando os labels aleatoriamente
        y = np.random.randint(0, n_classes, size=n_samples)
        return X, y


    def test_predict(self):
        X, y = self.generate_random_dataset(100, 10, 2)
        # Instanciar um objeto da classe DecisionTree
        dt = DecisionTree(max_depth=5)
        # Chamar o método fit
        dt.fit(X, y)
        # Chamar o o método predict
        y_pred = dt.predict(X)
        # Verificar se o vetor y_pred tem o mesmo tamanho que o vetor y
        self.assertEqual(len(y_pred), len(y))
        print("y:", y, "\ny_pred:", y_pred)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        print("Acurácia:", acc, "\nPrecisão:", prec)

def main():
    test_dt = TestDecisionTree()
    test_dt.test_predict()

if __name__ == "__main__":
    main()

