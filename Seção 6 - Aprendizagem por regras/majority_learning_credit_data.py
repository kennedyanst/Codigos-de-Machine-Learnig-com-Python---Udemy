import Orange

base_credit = Orange.data.Table("credit_data_regras.csv") #Colocar o "c#" na classe que será prevista na base de dados. Colocar i# no atributo q não será usado (clientid) 
base_credit.domain

majority = Orange.classification.MajorityLearner()
previsoes = Orange.evaluation.testing.TestOnTestData(base_credit, base_credit, [majority])
Orange.evaluation.CA(previsoes) #CA = Classification Accuracy

for registro in base_credit:
    print(registro.get_class())

from collections import Counter
Counter(str(registro.get_class()) for registro in base_credit)