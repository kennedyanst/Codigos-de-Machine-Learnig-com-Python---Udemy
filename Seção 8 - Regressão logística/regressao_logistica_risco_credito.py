import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression


with open("risco_credito.pkl", "rb") as f:
    X_risco_credito, y_risco_credito = pickle.load(f)

X_risco_credito

y_risco_credito # 2, 7, 11

#Apagando o risco MODERADO
X_risco_credito = np.delete(X_risco_credito, [2, 7, 11], axis = 0) #axis = linhas[0] e colunas[1] 
y_risco_credito = np.delete(y_risco_credito, [2,7,11], axis = 0)

logistic_risco_credito = LogisticRegression(random_state = 1)
logistic_risco_credito.fit(X_risco_credito, y_risco_credito)
logistic_risco_credito.intercept_ #B0 = -0,80
logistic_risco_credito.coef_ #B1 = -0.76, B2 = 0.23, B3 = -0.47, B4 = 1.12


#Previsoes do NAIVE BAYES. 
# historia boa (0), dívida alta (0), garantias nenhuma (1), renda > 35k (2)
#historia ruim (2), divida alta (0), garantias adequada (0), renda < 15k (0)
previsao1 = logistic_risco_credito.predict([[0,0,1,2], [2,0,0,0]])
