import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


base_casas = pd.read_csv("house_prices.csv")
base_casas.describe()
base_casas.isnull().sum() #Sem dados faltantes. 
base_casas.corr() #Calculando a correlação das variaveis. 

figura = plt.figure(figsize=(20,20))
sns.heatmap(base_casas.corr(), annot=True) #Mapa de calor das correlações. "annot = True: Mostrar os valores no mapa"
#Maiores correlação com o preço é a nota. (sqlt_living/price = 0.7, grade/price = 0.67)

X_casas = base_casas.iloc[:,5:6].values #Vai buscar apenas a coluna 5 (metragem)
y_casas = base_casas.iloc[:, 2].values #Selecionando a culna preços

from sklearn.model_selection import train_test_split
X_casas_treinamento, X_casas_teste, y_casas_treinamento, y_casas_teste = train_test_split(X_casas, y_casas, test_size = 0.3, random_state = 0) #Random_state = 0: Sempre ter a mesma divisão com os mesmos registros

X_casas_treinamento.shape,  y_casas_treinamento.shape
X_casas_teste.shape, y_casas_teste.shape

from sklearn.linear_model import LinearRegression
regressor_simples_casas = LinearRegression()
regressor_simples_casas.fit(X_casas_treinamento, y_casas_treinamento)


#b0
regressor_simples_casas.intercept_

#b1
regressor_simples_casas.coef_

regressor_simples_casas.score(X_casas_treinamento, y_casas_treinamento)
regressor_simples_casas.score(X_casas_teste, y_casas_teste)

previsoes = regressor_simples_casas.predict(X_casas_treinamento)

grafico = px.scatter(x= X_casas_treinamento.ravel(), y=previsoes)
grafico.show()


grafico1 = px.scatter(x=X_casas_treinamento.ravel(), y = y_casas_treinamento)
grafico2 = px.line(x = X_casas_treinamento.ravel(), y = previsoes)
grafico2.data[0].line.color = "red"
grafico3 = go.Figure(data= grafico1.data + grafico2.data)
grafico3

previsoes_teste = regressor_simples_casas.predict(X_casas_teste)

y_casas_teste

#Calculo do MAE (Mean absolut error)
abs(y_casas_teste - previsoes_teste).mean()
#172604.1288999542 para cima ou para baixo

from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(y_casas_teste, previsoes_teste) #172604.1288999542
mean_squared_error(y_casas_teste, previsoes_teste) #70170013932.1159
np.sqrt(mean_squared_error(y_casas_teste, previsoes_teste)) #264896.23238565685 (ROOT MEAN SQUARED ERROR)



grafico1 = px.scatter(x = X_casas_teste.ravel(), y = y_casas_teste)
grafico2 = px.line(x = X_casas_teste.ravel(), y = previsoes_teste)
grafico2.data[0].line.color = "red"
grafico3 = go.Figure(data= grafico1.data + grafico2.data)
grafico3