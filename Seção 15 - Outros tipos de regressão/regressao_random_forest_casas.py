from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np


base_casas = pd.read_csv("house_prices.csv")

X_casas = base_casas.iloc[:, 3:19].values
y_casas = base_casas.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(X_casas, y_casas, test_size=0.3, random_state=0)

X_casas_treinamento.shape
X_casas_teste.shape

regressor_random_forest_casas = RandomForestRegressor(n_estimators=200)
regressor_random_forest_casas.fit(X_casas_treinamento, y_casas_treinamento)
regressor_random_forest_casas.score(X_casas_treinamento, y_casas_treinamento)
regressor_random_forest_casas.score(X_casas_teste, y_casas_teste) #0.88

previsoes = regressor_random_forest_casas.predict(X_casas_teste)
y_casas_teste

mean_absolute_error(y_casas_teste, previsoes) #Erro de 67615K dols para cima ou para baixo em media 