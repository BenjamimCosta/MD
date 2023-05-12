import itertools
'''
Classe "TransactionDataset" que permite armazenar e acessar items, representando uma "Transactional database"
'''
class TransactionDataset:
    def __init__(self, transactions):
        # transactions é uma lista de listas, onde cada lista representa uma transação com os items comprados
        self.transactions = transactions
        # items é um conjunto com todos os items diferentes presentes nas transações
        self.items = set()
        for t in transactions:
            for i in t:
                self.items.add(i)

    def get_transactions(self):
        return self.transactions

    def get_items(self):
        return self.items
'''
Classe que implementa o algoritmo "Apriori". 
O construtor recebe um "TransactionDataset" e um suporte mínimo (o suporte mínimo serve para sabermos se um determinado conjunto de items é considerado frequente ou não).
'''
class Apriori:
    '''
    Neste código são usados frozensets porque são conjuntos imutáveis, ou seja, não podem ser modificados depois de criados.
    Aqui é importante porque usamos estes conjuntos como chaves de um dicionário que armazena os suportes dos candidatos.
    Se fossem usados conjuntos normais (mutáveis) seria possível haver uma alteração dos elementos dos conjuntos e isso afetaria o acesso aos valores do dicionário.
    '''
    def __init__(self, dataset, min_support):
        #transaction dataset
        self.dataset = dataset
        # suporte mínimo
        self.min_support = min_support
        # dicionário que armazena os conjuntos de items frequentes e os seus suportes
        self.frequent_itemsets = {}
        # lista que armazena as regras de associação
        self.rules = []

    def generate_candidates(self, k):
        # gera os conjuntos candidatos a items frequentes de tamanho k a partir dos conjuntos de items frequentes de tamanho k-1
        # se k = 1, gera os candidatos a partir dos items individuais do dataset
        candidates = []
        if k == 1:
            for i in self.dataset.get_items():
                candidates.append(frozenset([i]))
        else:
            # "prev_frequent" possui os conjuntos de elementos mais frequentes da iteração anterior
            prev_frequent = [f for f in self.frequent_itemsets if len(f) == k-1]
            # combinamos dois conjuntos de items frequentes de tamanho k-1 que tenham k-2 items em comum
            for i in range(len(prev_frequent)):
                for j in range(i+1, len(prev_frequent)):
                    common = prev_frequent[i].intersection(prev_frequent[j])
                    if len(common) == k-2:
                        candidate = prev_frequent[i].union(prev_frequent[j])
                        # verificamos se todos os subconjuntos de tamanho k-1 do candidato estão no conjunto de items frequentes de tamanho k-1 (pruning)
                        valid = True
                        for subset in itertools.combinations(candidate, k-1):
                            if frozenset(subset) not in prev_frequent:
                                valid = False
                                break
                        # se o candidato é válido, adiciona à lista de candidatos
                        if valid and candidate not in candidates:
                            candidates.append(candidate)
        return candidates

    def count_support(self, candidates):
        # conta o suporte dos candidatos a conjuntos de items frequentes
        counts = {}
        for t in self.dataset.get_transactions():
            for c in candidates:
                if c.issubset(t):
                    if c not in counts:
                        counts[c] = 1
                    else:
                        counts[c] += 1
        return counts

    def filter_candidates(self, counts):
        # remove os candidatos que tenham suporte inferior ao mínimo definido
        for c in counts:
            support = counts[c]
            if support >= self.min_support:
                self.frequent_itemsets[c] = support

    def generate_rules(self):
        # gera as regras de associação a partir dos conjuntos de items frequentes e os seus suportes
        for f in self.frequent_itemsets:
            if len(f) > 1:
                # para cada conjunto de items frequente com mais de um item, gera todas as possíveis regras do tipo X -> Y, onde X e Y são subconjuntos não vazios e disjuntos de f
                for i in range(1, len(f)):
                    # combinations dá um vetor de possiveis elementos de tamanho i de cada elemento mais frequente
                    for X in itertools.combinations(f, i):
                        X = frozenset(X)
                        Y = f.difference(X)
                        # calcula a confiança da regra como o suporte de "X U Y" dividido pelo suporte de X
                        confidence = self.frequent_itemsets[f] / self.frequent_itemsets[X]
                        # adiciona a regra à lista de regras se a confiança for maior ou igual ao mínimo definido
                        if confidence >= self.min_confidence:
                            self.rules.append((X, Y, confidence))

def main():
    transactions = [
        [1,3,4,6],
        [2,3,5],
        [1,2,3,5],
        [1,5,6]
    ]
    dataset = TransactionDataset(transactions)
    min_support = 2
    min_confidence = 0.6
    apriori = Apriori(dataset, min_support)
    apriori.min_confidence = min_confidence

    k = 1
    while True:
        # Geramos os candidatos da iteração atuals
        candidates = apriori.generate_candidates(k)
        # Se não houver mais candidatos chegamos ao fim do algoritmo
        if not candidates:
            break
        # Contamos o suporte de cada candidato
        counts = apriori.count_support(candidates)
        # Removemos os candidatos que não têm suporte mínimo
        apriori.filter_candidates(counts)
        # Incrementamos k para gerar os próximos candidatos
        k += 1

    # Geramos as regras de associação a partir dos conjuntos de items frequentes e seus suportes
    apriori.generate_rules()
    print("Dataset:")
    print(transactions)
    print("Conjuntos de items frequentes e os seus suportes:")
    for f in apriori.frequent_itemsets:
        print(f, apriori.frequent_itemsets[f])
    print("Regras de associação e as suas confianças:")
    for r in apriori.rules:
        print(r[0], "->", r[1], r[2])

if __name__ == "__main__":
    main()
