import numpy as np
#from sklearn.feature_selection import f_classif, f_regression
import sys
sys.path.append('../Aula1')
from dataset import Dataset
from f_classif import f_classif
from f_regression import f_regression

# Definir a classe SelectKBest
class SelectKBest():
    '''
    Classe que faz seleção/filtração de features de um Dataset tendo em conta o seu p_value (mantem as K features com valores mais baixos)

    Argumentos:
        - score_func -> Score function usada para filtrar as features
        - K : int -> Número de features a selecionar
    Parâmetros estimados:
        - F_ :  Array de tamanho n (para n features) -> Mede a variação entre as classes em relação à variação dentro das classes. 
                                                        Quanto maior o valor F, maior é a diferença entre as classes e mais relevante é a característica. 
        - p_ :  Array de tamanho n (para n features) -> Indicador do nível de significância estatística do valor F.
                                                        Representa a probabilidade de que a diferença observada entre as classes seja devida ao acaso e não à influência da característica.
                                                        Ou seja, quanto menor o valor p, menor é a probabilidade de que a característica seja irrelevante.
    '''
    def __init__(self, score_func=f_regression, k=2):
        self.score_func = score_func # função para a pontuação
        self.k = k # número de características a selecionar
    
    # Estimar os parâmetros F e p para cada característica usando a score_func
    def fit(self, X, y):
        # Garantir que X e y são arrays numpy
        #X = np.asarray(X)
        #y = np.asarray(y)
        # Calcular F e p para cada característica usando a score_func
        if self.score_func == f_classif:
            F, p = self.score_func(X, y)
            self.F_ = F
            self.p_ = p
        elif self.score_func == f_regression:
            p = self.score_func(X, y)
            self.p_ = p    

        return self
    
    # Selecionar as k características com o p_value mais baixo
    def transform(self, X):
        # Garantir que X é um array numpy
        #X = np.asarray(X)
        # Obter os índices das k características com o p_value mais baixo:
        k = min(self.k, X.shape[1]) # garantir que k não é maior que o número de características
        idx = np.argsort(self.p_)[:k] # ordenar os índices pelo p_value e selecionar os primeiros k
        # Retornar a matriz X com apenas as k características selecionadas
        return X[:, idx]
    
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
    

# Criar um conjunto de dados de teste
dataset = Dataset(X=np.array([[1, 3, 5, 7],
 [9, 5, 6, 8],
 [3, 200, 2, 1],
 [12, 6, 8, 6]]),
 y=np.array([0, 1, 0, 1]),
 features_names=["f1", "f2", "f3", "f4"],
 label="y")

X = dataset.X
y = dataset.y

# Criar uma instância do transformador SelectKBest
skb = SelectKBest()

#X_new = skb.fit_transform(X, y)
skb = skb.fit(X,y)
X_new = skb.transform(X)
print(X)
print("Forma da matriz original:", X.shape)
if skb.score_func == 'f_classif':
    print("Valores de F para cada característica:", skb.F_)
print("Valores de p para cada característica:", skb.p_)
print(X_new)
print("Forma da matriz transformada:", X_new.shape)
