import numpy as np
import sys
sys.path.insert(1, './Aula1')
from dataset import Dataset
from scipy import stats
'''
Função que usa o algoritmo F-statistics: "F = (R^2 / (1 - R^2)) * (n - 2)" para medir a relação entre a variância explicada e não explicada de cada característica em relação ao target.´
Em seguida, calcula o p-value correspondente usando a distribuição F.
'''
def f_regression(dataset: Dataset):
    X = dataset.X
    y = dataset.y
    
    n = X.shape[0]  # número de amostras
    k = X.shape[1]  # número de características
    
    R2 = np.zeros(k)  # vetor para armazenar o R^2 para cada característica
    F = np.zeros(k)   # vetor para armazenar o valor F para cada característica
    
    # Loop pelas características e calcular o R^2 e F para cada uma
    for i in range(k):
        x_i = X[:,i]
        # calcular R^2 para a i-ésima característica
        b1, b0 = np.polyfit(x_i, y, 1)
        y_hat = b1 * x_i + b0
        ss_res = np.sum((y - y_hat)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        R2[i] = 1 - ss_res/ss_tot
        # calcular F para a i-ésima característica
        F[i] = (R2[i] / (1 - R2[i])) * (n - 2)
    
    # Calcular os p-values usando a distribuição F
    p_values = 1 - stats.f.cdf(F, 1, n - 2)
    
    return p_values
