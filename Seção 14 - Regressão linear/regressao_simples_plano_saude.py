import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


base_plano_saude = pd.read_csv("plano_saude.csv")

X_plano_saude =  base_plano_saude.iloc[:, 0].values
y_plano_saude = base_plano_saude.iloc[:, 1].values

np.corrcoef(X_plano_saude, y_plano_saude) #CORRELAÇÃO DE 93%
X_plano_saude.shape #Uma dimensão
X_plano_saude = X_plano_saude.reshape(-1,1)
X_plano_saude.shape #Matriz

from sklearn.linear_model import LinearRegression
regressor_plano_saude = LinearRegression()
regressor_plano_saude.fit(X_plano_saude, y_plano_saude)

#b0 = Inicio da linha de regressão
regressor_plano_saude.intercept_

#b1 = Declive da linha 
regressor_plano_saude.coef_

previsoes = regressor_plano_saude.predict(X_plano_saude) #Previsões são geradas pela aplicação da formula da reta com a função "predict()": Y = b0 + b1 . x1


X_plano_saude.ravel() #Precisa voltar a ser vetor para ir no grafico. O grafico não aceita matriz
grafico = px.scatter(x = X_plano_saude.ravel(), y = y_plano_saude)
grafico.add_scatter(x = X_plano_saude.ravel(), y = previsoes, name = "Regressão" )
grafico.show()


regressor_plano_saude.score(X_plano_saude, y_plano_saude) #Qualidade do algoritmo. Quanto mais proximo de 1, melhor!

#Mostra os residuais. O quão afastado os graficos estão das linha de regressão. Train R² significa a qualidade do algoritmo
from yellowbrick.regressor import ResidualsPlot
visualizador = ResidualsPlot(regressor_plano_saude)
visualizador.fit(X_plano_saude, y_plano_saude)
visualizador.poof()
