import pandas as pd
from apyori import apriori 

base_mercado1 = pd.read_csv("mercado1.csv", header= None)

#Colocar os dados em formato de lista
transacoes = []

for i in range (len(base_mercado1)):
    print(base_mercado1.values[i, 0])
    transacoes.append([str(base_mercado1.values[i, j]) for j in range(base_mercado1.shape[1])])

type(transacoes)

# ----- A BIBLIOTECA APYORI REQUER QUE OS DADOS ESTEJAM EM UM FORMATO DE LISTA 

# ------ Gerando as regras de associação
regras = apriori(transacoes, min_support = 0.3, min_confidence = 0.8, min_lift = 2)
resultados = list(regras)

len(resultados)
resultados[0]
resultados[1]
resultados[2]
# RelationRecord(items=frozenset({'cafe', 'manteiga', 'pao'}), support=0.3, ordered_statistics=
# [OrderedStatistic(items_base=frozenset({'cafe'}), items_add=frozenset({'manteiga', 'pao'}), confidence=1.0, lift=2.5), 
# OrderedStatistic(items_base=frozenset({'cafe', 'manteiga'}), items_add=frozenset({'pao'}), confidence=1.0, lift=2.0), 
# OrderedStatistic(items_base=frozenset({'cafe', 'pao'}), items_add=frozenset({'manteiga'}), confidence=1.0, lift=2.0)])

#OrderedStatistic = Regra.

resultados[2][0]

r = resultados[2][2]
r[0] #Regra 1
r[1] #Regra 2
r[2] #Regra 3
r[2][0] #Retorna a parte SE
r[2][1] #Retorna a parte ENTÃO 
r[2][2] #Confiança 
r[2][3] #Valor do lift


# -------- Construindo uma codificação para VISUALIZAR AS REGRAS usando o loop "for"
A = []
B = []
suport = []
confianca = []
lift = []

for resultado in resultados:
    #print(resultado)
    s = resultado[1] #suporte
    result_rules = resultado[2] #resultado das regras
    for result_rule in result_rules:
        #print(result_rule)
        a = list(result_rule[0])
        b = list(result_rule[1])
        c = result_rule[2]
        l = result_rule[3]
        #print(a, " - ", b, " - ", c, " - ", l)
        A.append(a)
        B.append(b)
        suport.append(s)
        confianca.append(c)
        lift.append(l)

A
B
suport
confianca
lift

rules_df = pd.DataFrame({"A": A, "B": B, "Confiança": confianca, "Suporte": suport, "Lift": lift})
rules_df.sort_values(by = "Lift", ascending=False) #Ordenando as regras de forma decrescente pelo lift

