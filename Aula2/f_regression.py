import numpy as np
from scipy.stats import linregress

# Definir a função f_regression
def f_regression(X, y):
    # Garantir que X e y são arrays numpy
    #X = np.asarray(X)
    #y = np.asarray(y)
    # Obter o número de features
    n_features = X.shape[1]
    # Inicializar o array para armazenar os valores de p
    p = np.zeros(n_features)
    # Para cada feature:
    for i in range(n_features):
        # Aplicar a função linregress
        slope, intercept, r_value, p_value, std_err = linregress(X[:, i], y)
        p[i] = p_value
    return p
