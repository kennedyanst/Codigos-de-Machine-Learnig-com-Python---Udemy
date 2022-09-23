import Orange

base_census = Orange.data.Table("census_regras.csv") #Colocar o "c#" na classe que será prevista na base de dados. Colocar i# no atributo q não será usado (clientid) 
base_census.domain

#ZeroR
majority = Orange.classification.MajorityLearner()
previsoes = Orange.evaluation.testing.TestOnTestData(base_census, base_census, [majority])
Orange.evaluation.CA(previsoes) #O algoritmo precisa acertar mais do que esse valor. 

from collections import Counter
Counter(str(registro.get_class()) for registro in base_census)
#linha base = 22654 / (22654 + 7508) # = 75,10