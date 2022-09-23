import Orange

base_credit = Orange.data.Table("credit_data_regras.csv") #Colocar o "c#" na classe que será prevista na base de dados. Colocar i# no atributo q não será usado (clientid) 
base_credit.domain

#Divisão da base que será usado no treinamento e teste
base_dividida = Orange.evaluation.testing.sample(base_credit, n = 0.25)
base_dividida[0]
base_dividida[1]

#Base de treinamento e teste
base_treinamento = base_dividida[1]
base_teste = base_dividida[0]

len(base_treinamento), len(base_teste)

#Função de machine learning de regras
cn2 = Orange.classification.rules.CN2Learner()
regras_credit = cn2(base_treinamento)

#Geração das regras
for regras in regras_credit.rule_list:
    print(regras)
    # IF age>=34.9257164876908 THEN default=0 
    # IF loan<=2495.13299137587 AND income>=20145.9885970689 THEN default=0 
    # IF income<=31702.3342987522 AND loan>=3665.88089899456 THEN default=1 
    # IF loan>=7329.243163822591 AND loan>=9601.375482171099 THEN default=1 
    # IF loan>=7329.243163822591 AND loan>=9595.28628892989 THEN default=0 
    # IF loan>=7329.243163822591 AND age>=29.2626339835115 THEN default=1 
    # IF loan>=7498.630446855849 AND age>=21.4227129220963 THEN default=1 
    # IF age>=33.8957485635765 AND income>=34237.5754192472 THEN default=0 
    # IF age<=18.1760434475727 AND age>=18.1760434475727 THEN default=1 
    # IF age<=18.3097456344403 AND income>=52981.508597731605 THEN default=0 
    # IF loan>=5473.98555060076 AND age>=33.6895613595843 THEN default=1 
    # IF age>=33.5187431662343 THEN default=0 
    # IF loan<=5502.73603087282 AND age>=33.2805235567503 THEN default=1 
    # IF loan<=5502.73603087282 AND age>=30.5151092219166 THEN default=0 
    # IF loan<=5502.73603087282 AND income>=40496.2558229454 THEN default=0 
    # IF income>=60019.447135273396 AND age>=20.3008601283655 THEN default=0 
    # IF loan>=6043.14310633161 AND age>=26.854012909811 THEN default=1 
    # IF income<=22089.8374845274 AND age>=21.3656869572587 THEN default=1 
    # IF loan<=4285.38691174949 AND income>=33489.0398592688 THEN default=0 
    # IF loan>=6994.48780081424 THEN default=1 
    # IF loan<=3105.4430213977303 AND loan>=3105.4430213977303 THEN default=1 
    # IF loan<=3343.81635769923 AND income>=27407.056202646298 THEN default=0 
    # IF age>=23.4877088945359 AND loan>=6289.25607587104 THEN default=1 
    # IF income>=48790.1324336417 AND income>=50527.5841732509 THEN default=0 
    # IF loan>=4625.19337762744 AND loan>=5862.83302915672 THEN default=1 
    # IF income>=48430.3596126847 THEN default=0 
    # IF loan>=4625.19337762744 THEN default=1 
    # IF age>=30.142012033611497 THEN default=1 
    # IF income>=31702.3342987522 THEN default=0 
    # IF income>=26218.4948474169 THEN default=1 
    # IF income<=24857.6948815025 THEN default=0 
    # IF income<=25146.5956843458 THEN default=0 
    # IF TRUE THEN default=0 

#Resultado
previsoes = Orange.evaluation.testing.TestOnTestData(base_treinamento, base_teste, [lambda testdata: regras_credit])
Orange.evaluation.CA(previsoes)
# 97,2%
