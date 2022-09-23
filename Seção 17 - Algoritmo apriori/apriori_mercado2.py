import pandas as pd 
import numpy as np
from apyori import apriori


base_mercado2 = pd.read_csv("mercado2.csv", header = None)


transacoes = []
for i in range(base_mercado2.shape[0]):
    transacoes.append([str(base_mercado2.values[i, j]) for j in range(base_mercado2.shape[1])])

# Produtos que são vendidos 4 vezes por dia
#4 * 7 = 28
#28/7501 = 0.003732 ---> Esse será o valor do suporte utilizado, produtos que são vendidos 4x ao dia

# Base de dados grandes, dificilmente o valor da confiança será alto



regras = apriori(transacoes, min_support = 0.003, min_confidence = 0.2, min_lift = 3)
resultados = list(regras)
len(resultados)

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
rules_df.sort_values(by = "Confiança", ascending=False) #Ordenando as regras de forma decrescente pelo lift
