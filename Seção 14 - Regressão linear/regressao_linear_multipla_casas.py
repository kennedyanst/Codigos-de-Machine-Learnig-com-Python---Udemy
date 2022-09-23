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

X_casas_treinamento.shape, X_casas_teste.shape


from sklearn.linear_model import LinearRegression
regressor_multiplo_casas = LinearRegression()
regressor_multiplo_casas.fit(X_casas_treinamento, y_casas_treinamento)

regressor_multiplo_casas.intercept_
regressor_multiplo_casas.coef_

regressor_multiplo_casas.score(X_casas_treinamento, y_casas_treinamento)
regressor_multiplo_casas.score(X_casas_teste, y_casas_teste)

previsoes = regressor_multiplo_casas.predict(X_casas_teste)

from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(y_casas_teste, previsoes) #Erro de 123k para cima ou para baixo. 
mean_squared_error(y_casas_teste, previsoes)

