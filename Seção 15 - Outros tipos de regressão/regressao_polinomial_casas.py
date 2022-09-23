from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


base_casas = pd.read_csv("house_prices.csv")

X_casas = base_casas.iloc[:, 3:19].values
y_casas = base_casas.iloc[:, 2].values

from sklearn.model_selection import train_test_split
X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(X_casas, y_casas, test_size=0.3, random_state=0)


X_casas_treinamento.shape
X_casas_teste.shape

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 2)
X_casas_treinamento_poly = poly.fit_transform(X_casas_treinamento)
X_casas_teste_poly = poly.transform(X_casas_teste)

X_casas_treinamento_poly.shape, X_casas_teste_poly.shape

regressor_casas_poly = LinearRegression()
regressor_casas_poly.fit(X_casas_treinamento_poly, y_casas_treinamento)

regressor_casas_poly.score(X_casas_treinamento_poly, y_casas_treinamento) #GANHO DE: 0.8179320662222848
regressor_casas_poly.score(X_casas_teste_poly, y_casas_teste) #GANHO DE: 0.815308613608555

previsoes = regressor_casas_poly.predict(X_casas_teste_poly)
y_casas_teste

from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(y_casas_teste, previsoes) #101K de erro para cima ou para baixo 