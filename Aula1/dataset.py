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
            
    def read_csv(self, filename):
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
            self.X = np.array(data)[:, :-1]
            self.y = np.array(data)[:, -1]
        except FileNotFoundError:
            print(f'Ficheiro "{filename}" não encontrado.')

   
    def write_csv(self, filename):
        try:
            with open(filename, 'w') as f:
                # escreve os nomes das colunas na primeira linha do ficheiro
                f.write(','.join(self.features_names) + ',' + self.label + '\n')
                # escreve os dados do dataset nas linhas seguintes:
                for row in range(self.X.shape[0]):
                    #cria string com os valores de cada coluna da linha atual, separados por vírgulas:
                    #join() concatena uma lista de strings usando vírgula como separador 
                    #a lista de strings é criada iterando sobre cada elemento da linha atual e convertendo-o em string
                    f.write(','.join([str(elem) for elem in self.X[row]]) + ',' + str(self.y[row]) + '\n')
        except IOError:
            print(f'Erro ao escrever o ficheiro "{filename}"')

    def read_tsv(self, file, label=None):
        #só muda o separador
        self.read_csv(file, label, '\t')

    def write_tsv(self, filename):
        try:
            with open(filename, 'w') as f:
                # Escreve os nomes das colunas do conjunto de dados
                header = '\t'.join(self.features_names)
                header += '\t'+self.label+'\n'
                f.write(header)

                # Escreve as linhas do conjunto de dados
                for row in range(self.X.shape[0]):
                    row_values = '\t'.join([str(elem) for elem in self.X[row]])
                    row_values += '\t' + str(self.y[row]) + '\n'
                    f.write(row_values)
        except IOError:
            print("Erro ao escrever o ficheiro: " + filename)

    def count_nulls(self):
        x_nulls = np.sum(self.X == '', axis=0)
        return x_nulls

    def fill_nulls(self):
 
        # Função que verifica se uma coluna pode ser convertida em float
        def is_floatable(col):
            try:
                col.astype(float)
                return True
            except ValueError:
                return False

        # Substituir valores vazios por NaN 
        self.X[self.X == ''] = np.nan

        # Ver se cada coluna pode ser convertida em float para distinguir numéricas de categóricas
        floatable = np.apply_along_axis(is_floatable, axis=0, arr=self.X)

        # Selecionar apenas as colunas numéricas
        X_numeric = self.X[:, floatable]
        
        # Calcular as médias das linhas com valores numéricos, ignorando os valores NaN
        means = np.nanmean(X_numeric.astype(float), axis=1)
        # Substituir os valores vazios (NaN) das colunas pela média correspondente ou por 'empty' conforme seja numerico ou categorico
        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                if self.X[i][j] == 'nan' and floatable[j] :
                    self.X[i][j] = means[i]
                elif self.X[i][j] == 'nan' and not floatable[j] :
                    self.X[i][j] = 'empty'
        
        print('Valores vazios substituídos.')



    def describe(self):
        # Calcula o número de elementos, média, desvio padrão, valor mínimo, valor máximo e quartis para cada coluna numérica:

        num_elements = self.X.shape[0]
        
        numeric_cols = []  # Lista para guardar as colunas numéricas
        categorical_cols = []  # Lista para guardar as colunas categóricas
        
        for i in range(self.X.shape[1]):  # Itera sobre as colunas 
            try:  # Tenta converter a coluna para float
                float_col = self.X[:, i].astype(float)
                numeric_cols.append(i)  # Se não der erro, adiciona o índice da coluna à lista de colunas numéricas
                
            except ValueError:  # Se der erro ao converter para float é pq é categórica
                categorical_cols.append(i)
        
        means = np.mean(self.X[:, numeric_cols].astype(float), axis=0)
        stds = np.std(self.X[:, numeric_cols].astype(float), axis=0)
        mins = np.min(self.X[:, numeric_cols].astype(float), axis=0)
        maxs = np.max(self.X[:, numeric_cols].astype(float), axis=0)
        p25 = np.percentile(self.X[:, numeric_cols].astype(float), 25, axis=0)
        p50 = np.percentile(self.X[:, numeric_cols].astype(float), 50, axis=0)
        p75 = np.percentile(self.X[:, numeric_cols].astype(float), 75, axis=0)

        # Imprime os resultados para cada coluna numérica
        for i in range(len(numeric_cols)):
            feature_name = self.features_names[numeric_cols[i]]
            print("\n")
            print("Feature:", feature_name)
            print("Número de elementos:", num_elements)
            print("Valor mínimo:", mins[i])
            print("Valor máximo: ", maxs[i])
            print("Média:", means[i])
            print("Desvio padrão:", stds[i])
            print("1º Quartil:", p25[i])
            print("2º Quartil (mediana):", p50[i])
            print("3º Quartil:", p75[i])     
        
        # Calcula a frequência de cada valor para as colunas categóricas
        for i in range(len(categorical_cols)):
            feature_name = self.features_names[categorical_cols[i]]
            values, counts = np.unique(self.X[:, categorical_cols[i]], return_counts=True)  # Obtém os valores únicos e as suas contagens
            print("\n")
            print("Feature:", feature_name)
            print("Number of elements:", num_elements)
            for j in range(len(values)):  # Imprime cada valor e a sua frequência
                print(f"{values[j]}: {counts[j]} ({counts[j]/num_elements*100:.2f}%)")

ds = Dataset()

# Lê o ficheiro "notas.csv" e armazena os dados em X e y
ds.read_csv("notas.csv")

# Imprime os nomes das features, a matriz X, o vetor y e o nome da label
print("Nomes das features:\n" + str(ds.features_names))
print("X:\n" + str(ds.X))
print("y:\n" + str(ds.y))
print("Label:\n" + str(ds.label))

# Conta e imprime o número de valores vazios em cada coluna
print("Número de valores nulos:\n" + str(ds.count_nulls()))

# Preenche os valores vazios com as médias nas colunas numéricas ou com a string 'empty' em colunas categóricas
ds.fill_nulls()

# Imprime a matriz X e o vetor y após preencher os valores vazios
print("X:\n" + str(ds.X))
print("y:\n" + str(ds.y))

# Descreve as estatísticas de cada coluna
ds.describe()

# Testar funções de escrita para ficheiro
ds.write_csv("teste.csv")
ds.write_tsv("teste2.tsv")

