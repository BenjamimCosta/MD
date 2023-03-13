import numpy as np
from Aula1.dataset import Dataset
from typing import Callable, Tuple
import numpy as np

class SelectKBest:
    def __init__(self, score_func: Callable, k: int):
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None
        
    #estima os valores F e p para cada feature usando a função score_func.
    def fit(self, dataset : Dataset) -> None:
        self.F, self.p = self.score_func(dataset)
        return None
    
    #seleciona as k features com o p-value mais baixo. O método retorna o dataset com as k melhores features selecionadas.
    def transform(self, dataset : Dataset) -> np.ndarray:
        indices = np.argsort(self.p)[:self.k]
        features = np.array(dataset.features_names)[indices]
        return Dataset(dataset.X[:, indices], dataset.get_y, list(features), dataset.get_label)
    
    def fit_transform(self, dataset : Dataset) -> np.ndarray:
        self.fit(dataset)
        return self.transform(dataset)
