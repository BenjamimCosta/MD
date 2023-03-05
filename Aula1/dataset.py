import csv
import numpy as np

class Dataset:

    #X -> matriz com as variáveis de entrada (pensar como tratar as variáveis de tipos distintos)
    #y -> vetor com a saída
    #features_names -> vetor com o nome das features
    #label -> string com o nome da label (atributo saída)

    def __init__(self, X=None, y=None, features_names=None, label=None):
        self.X = X
        self.y = y
        self.features_names = features_names
        self.label = label
        
    def set_X(self, X):
        self.X = X
        
    def set_y(self, y):
        self.y = y
        
    def set_features_names(self, features_names):
        self.features_names = features_names
    
    def set_label(self, label):
        self.y = label
        
    def get_X(self):
        return self.X
    
    def get_y(self):
        return self.y
    
    def get_features_names(self):
        return self.features_names
    
    def get_label(self):
        return self.label
            
    def read_csv(self, filename, sep=','):
        try:
            #abrir o ficheiro CSV em modo leitura, with é usado para garantir que o ficheiro é fechado automaticamente depois da leitura
            with open(filename, 'r') as f:
                #armazenar as linhas na lista 'lines'
                lines = f.readlines()
                #strip() é usado para remover caracteres em branco do início e do fim de cada linha
                #split(',') divide cada linha numa lista de valores separados por vírgulas 
                #assim, 'data' guarda os dados do ficheiro CSV numa matriz, excluindo a primeira linha (que contém os nomes das colunas)
                data = [line.strip().split(',') for line in lines[1:]]
                #guardar os nomes das colunas numa lista, exceto o nome da última que é o nome da variável de saída
                self.features_names = lines[0].strip().split(',')[:-1]
                #guardar nome da variável de saída
                self.label =  lines[0].strip().split(',')[-1]
            #definir X e y:
            #se não houver variável de saída definida, então os dados armazenados em 'data'' são armazenados na variável de entrada X
            #caso contrário, os dados são divididos numa matriz de entrada X e um vetor de saída y
            if self.y is None:
                self.X = np.array(data)
            #[:, :-1] retorna todas as colunas da matriz, exceto a última (que contém os valores da variável de saída)
            else:
                self.X = np.array(data)[:, :-1]
                self.y = np.array(data)[:, -1]
        except FileNotFoundError:
            print(f'Arquivo "{filename}" não encontrado.')

    def write_csv(self, filename):
        try:
            with open(filename, 'w') as f:
                # escreve os nomes das colunas na primeira linha do arquivo
                f.write(','.join(self.features_names) + ',' + self.label + '\n')
                # escreve os dados do dataset nas linhas seguintes:
                #shape[0] diz o nº de linhas q a matriz X tem
                for row in range(self.X.shape[0]):
                    #cria string com os valores de cada coluna da linha atual, separados por vírgulas:
                    #join() concatena uma lista de strings usando vírgula como separador 
                    #a lista de strings é criada iterando sobre cada elemento da linha atual e convertendo-o em string
                    f.write(','.join([str(elem) for elem in self.X[row]]) + ',' + str(self.y[row]) + '\n')
        except IOError:
            print(f'Erro ao escrever o arquivo "{filename}"')

    def read_tsv(self, file, label=None):
        #só muda o separador
        self.read_csv(file, label, '\t')

    def write_tsv(X, y, filename):
        try:
            with open(filename, 'w') as f:
                # Escreve as colunas do conjunto de dados
                header = '\t'.join(['col' + str(i) for i in range(X.shape[1])])
                header += '\tlabel\n'
                f.write(header)

                # Escreve as linhas do conjunto de dados
                for row in range(X.shape[0]):
                    row_values = '\t'.join([str(elem) for elem in X[row]])
                    row_values += '\t' + str(y[row]) + '\n'
                    f.write(row_values)
        except IOError:
            print("Erro ao escrever o arquivo: " + filename)

    def count_nulls(self):
        x_nulls = np.sum(self.X == '', axis=0)
        return x_nulls

    def fill_nulls(self):
        # Ignorar a primeira coluna
        X_no_first = self.X[:, 1:]
        
        # Substituir valores vazios por NaN
        X_no_first[X_no_first == ''] = np.nan
        
        # Calcular as médias das notas de cada aluno ignorando os valores NaN
        means = np.nanmean(X_no_first.astype(float), axis=1)
        # Substituir os valores vazios das colunas pela média correspondente
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                if self.X[i][j] == 'nan':
                    self.X[i][j] = means[i]
        
        print('Valores vazios substituídos.')


    def describe(self):
        # Calcula o número de elementos, média, desvio padrão, valor mínimo, valor máximo e quartis para cada coluna
        num_elements = self.X.shape[0]
        means = np.mean(self.X[:, 1:].astype(float), axis=0)
        stds = np.std(self.X[:, 1:].astype(float), axis=0)
        mins = np.min(self.X[:, 1:].astype(float), axis=0)
        maxs = np.max(self.X[:, 1:].astype(float), axis=0)
        q25 = np.percentile(self.X[:, 1:].astype(float), 25, axis=0)
        q50 = np.percentile(self.X[:, 1:].astype(float), 50, axis=0)
        q75 = np.percentile(self.X[:, 1:].astype(float), 75, axis=0)

        # Imprime os resultados para cada coluna
        for i, feature_name in enumerate(self.features_names[1:]):
            if feature_name != "":
                print("\n")
                print("Feature:", feature_name)
                print("Number of elements:", num_elements)
                print("Minimum value:", mins[i])
                print("Maximum value: ", maxs[i])
                print("Mean:", means[i])
                print("Standard deviation:", stds[i])
                print("25th percentile:", q25[i])
                print("50th percentile (median):", q50[i])
                print("75th percentile:", q75[i])

ds = Dataset()
ds.read_csv('notas.csv')
print("features names:\n" + str(ds.features_names))
print("X:\n" + str(ds.X))
print("y:\n" + str(ds.y))
print("label name:\n" + str(ds.label)) 
print("count_nulls:\n" + str(ds.count_nulls()))
print("------------------------------")
ds.fill_nulls()
print("------------------------------")
print("X:\n" + str(ds.X))
print("y:\n" + str(ds.y))
print("count_nulls:\n" + str(ds.count_nulls()))
ds.describe()