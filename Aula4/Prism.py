class Prism:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.rules = []
    
    def fit(self):
      data = self.data.copy()
      target = self.target.copy()
      
      while len(data) > 0:
          # Encontra a classe mais commum
          most_common_class = target.value_counts().idxmax()
          
          # Encontra a melhor regra para a classe mais comum
          best_rule = self.find_best_rule(data, target, most_common_class)
          
          # Adiciona a melhor regra à lista de regras
          self.rules.append(best_rule)
          
          # Remove instâncias abrangidas pela melhor regra
          data = data[~best_rule(data)]
          target = target.loc[data.index]

    # Encontra a melhor regra para uma certa classe
    def find_best_rule(self, data, target, most_common_class):
        best_accuracy = 0
        best_rule = None
        
        # Itera sobre todos as features
        for feature in data.columns:
            # Itera apenas os únicos
            for value in data[feature].unique():
                # Cria uma regra que seleciona as instâncias
                rule = lambda x: x[feature] == value
                
                # Calcula a precisão da regra
                accuracy = (target[rule(data)] == most_common_class).mean()
                
                # Atualiza a melhor regra
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_rule = rule
        
        return best_rule
    def predict(self, data):
      predictions = []
      
      # Itera sobre todas as instâncias
      for i in range(len(data)):
          instance = data.iloc[i]
          
          # Itera todas as regras
          for rule in self.rules:
              # Verifiqua se a instância está coberta pela regra
              if rule(instance.to_frame().T).any():
                  predictions.append(self.target[rule(self.data)].mode()[0])
                  break
      
      return predictions


    # Print das regras
    def __repr__(self):
      rules_str = []
      for i, rule in enumerate(self.rules):
          feature = rule.__closure__[0].cell_contents
          value = rule.__closure__[1].cell_contents
          class_label = self.target[rule(self.data)].mode()[0]
          rules_str.append(f'Rule {i + 1}: If {feature} == {value}, then class = {class_label}')
      return '\n'.join(rules_str)


import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
target = pd.Series(iris.target)


classifier = Prism(data, target)

classifier.fit()
print(classifier)

predictions = classifier.predict(data)

accuracy = (predictions == target).mean()

print(f'Accuracy: {accuracy:.2f}')