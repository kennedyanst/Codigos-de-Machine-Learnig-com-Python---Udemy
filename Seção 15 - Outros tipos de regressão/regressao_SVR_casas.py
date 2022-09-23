from sklearn.svm import SVR
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

from sklearn.preprocessing import StandardScaler
scaler_x_casas = StandardScaler()
X_casas_treinamento_scaled = scaler_x_casas.fit_transform(X_casas_treinamento)
scaler_y_casas = StandardScaler()
y_casas_treinamento_scaled = scaler_y_casas.fit_transform(y_casas_treinamento.reshape(-1,1))

X_casas_teste_scaled = scaler_x_casas.transform(X_casas_teste)
y_casas_teste_scaled = scaler_y_casas.transform(y_casas_teste.reshape(-1,1))


regressor_svr_casas = SVR(kernel = "rbf")
regressor_svr_casas.fit(X_casas_treinamento_scaled, y_casas_treinamento_scaled.ravel())

regressor_svr_casas.score(X_casas_treinamento_scaled, y_casas_treinamento_scaled)
regressor_svr_casas.score(X_casas_teste_scaled, y_casas_teste_scaled)

previsoes = regressor_svr_casas.predict(X_casas_teste_scaled)
y_casas_teste_scaled

y_casas_teste_inverse = scaler_y_casas.inverse_transform(y_casas_teste_scaled)
previsoes_inverse = scaler_y_casas.inverse_transform(previsoes.reshape(-1,1))

mean_absolute_error(y_casas_teste_inverse, previsoes_inverse) 
#82453.02 dols ERRO para MAIS ou MENOS