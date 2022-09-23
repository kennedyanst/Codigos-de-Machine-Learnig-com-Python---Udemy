
import Orange

base_risco_credito = Orange.data.Table("risco_credito_regras.csv") #Colocar o "c#" na classe na ase de dados 
base_risco_credito.domain

cn2 = Orange.classification.rules.CN2Learner()
regras_risco_credito = cn2(base_risco_credito)

for regras in regras_risco_credito.rule_list:
    print(regras)
    # IF renda==0_15 THEN risco=alto 
    # IF historia==boa AND divida!=alta THEN risco=baixo 
    # IF historia==boa AND garantias!=nenhuma THEN risco=baixo 
    # IF historia==boa AND renda!=15_35 THEN risco=baixo 
    # IF historia==boa THEN risco=moderado 
    # IF divida==alta THEN risco=alto 
    # IF historia!=desconhecida THEN risco=moderado 
    # IF garantias==adequada THEN risco=baixo 
    # IF renda==15_35 THEN risco=moderado 
    # IF historia==desconhecida THEN risco=baixo 
    # IF TRUE THEN risco=alto 

previsoes = regras_risco_credito([["boa", "alta", "nenhuma", "acima_35"], ["ruim", "alta", "adequada", "0_15"]])
base_risco_credito.domain.class_var.values

for i in previsoes:
    #print(i)
    print(base_risco_credito.domain.class_var.values[i])
