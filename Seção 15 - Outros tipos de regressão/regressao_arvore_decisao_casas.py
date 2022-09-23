from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np



base_casas = pd.read_csv("house_prices.csv")

X_casas = base_casas.iloc[:, 3:19].values
y_casas = base_casas.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(X_casas, y_casas, test_size=0.3, random_state=0)


X_casas_treinamento.shape
X_casas_teste.shape

regressor_arvore_casas = DecisionTreeRegressor()
regressor_arvore_casas.fit(X_casas_treinamento, y_casas_treinamento)


regressor_arvore_casas.score(X_casas_treinamento, y_casas_treinamento)
regressor_arvore_casas.score(X_casas_teste, y_casas_teste) #Precisão de 0.70

previsoes = regressor_arvore_casas.predict(X_casas_teste) #Previsões do algoritmo
y_casas_teste #Respostas reais da base de dados

mean_absolute_error(y_casas_teste, previsoes) #99K de erro para cima ou para baixo
