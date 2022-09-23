import pandas as pd
import plotly.express as px
import numpy as np


base_plano_saude2 = pd.read_csv("plano_saude2.csv")

X_plano_saude2 = base_plano_saude2.iloc[:, 0:1].values #idade
y_plano_saude2 = base_plano_saude2.iloc[:, 1].values #custo do plano de saúde

from sklearn.tree import DecisionTreeRegressor
regressor_arvore_saude = DecisionTreeRegressor()
regressor_arvore_saude.fit(X_plano_saude2, y_plano_saude2)

previsoes = regressor_arvore_saude.predict(X_plano_saude2)

regressor_arvore_saude.score(X_plano_saude2, y_plano_saude2)

grafico = px.scatter(x = X_plano_saude2.ravel(), y = y_plano_saude2)
grafico.add_scatter(x= X_plano_saude2.ravel(), y = previsoes, name = "Regressão")
grafico.show()

X_teste_arvore = np.arange(min(X_plano_saude2), max(X_plano_saude2), 0.1) #Simulação de novas idades. 

X_teste_arvore = X_teste_arvore.reshape(-1,1) #Tranformando os dados em formato de matriz


grafico = px.scatter(x = X_plano_saude2.ravel(), y = y_plano_saude2)
grafico.add_scatter(x= X_teste_arvore.ravel(), y = regressor_arvore_saude.predict(X_teste_arvore), name = "Regressão")
grafico.show()

regressor_arvore_saude.predict([[40.6]])
