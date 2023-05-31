import numpy as np
from scipy.stats import f_oneway

def f_classif(X, y):
    # Garantir que X e y são arrays numpy
    #X = np.asarray(X)
    #y = np.asarray(y)
    # Obter o número de features e o número de classes
    n_features = X.shape[1]
    n_classes = len(np.unique(y))
    # Inicializar os arrays para armazenar os valores de F e p
    F = np.zeros(n_features)
    p = np.zeros(n_features)
    # Para cada feature:
    for i in range(n_features):
        # Agrupar as amostras pela classe
        groups = [X[y == c, i] for c in range(n_classes)]
        # Aplicar a função f_oneway aos grupos
        F[i], p[i] = f_oneway(*groups)
    return F, p
