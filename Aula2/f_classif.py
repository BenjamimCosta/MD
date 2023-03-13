import numpy as np
import sys
sys.path.insert(1, './Aula1')
from dataset import Dataset
from scipy.stats import f_oneway

def f_classif(dataset : Dataset):
    # Agrupar as samples/exemplos por classes tendo em conta os valores do target.
    classes = np.unique(dataset.get_y())
    grouped_features_values = [dataset.get_X()[dataset.get_y() == classe] for classe in classes]
    print(grouped_features_values)
    # Aplicar ANOVA aos elementos das diferentes classes para analisar a variância (diferença da média) entre essas classes, por coluna.
    # Dá nan quando todos os elementos são iguais, visto que não há variância.
    num_columns = dataset.X.shape[1]
    f_values, p_values = [],[]
    for i in range(num_columns):
        f, p = f_oneway(*[X[:, i] for X in grouped_features_values])
        f_values.append(f)
        p_values.append(p)

    return f_values,p_values

