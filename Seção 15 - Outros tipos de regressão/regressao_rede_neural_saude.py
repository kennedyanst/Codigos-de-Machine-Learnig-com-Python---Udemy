import pandas as pd
import plotly.express as px
import numpy as np


base_plano_saude2 = pd.read_csv("plano_saude2.csv")

X_plano_saude2 = base_plano_saude2.iloc[:, 0:1].values #idade
y_plano_saude2 = base_plano_saude2.iloc[:, 1].values #custo do plano de saúde

from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_plano_saude2_scaled = scaler_X.fit_transform(X_plano_saude2)
scaler_y = StandardScaler()
y_plano_saude2_scaled = scaler_X.fit_transform(y_plano_saude2.reshape(-1,1))


from sklearn.neural_network import MLPRegressor
regressor_rna_saude = MLPRegressor(max_iter=1000)
regressor_rna_saude.fit(X_plano_saude2_scaled, y_plano_saude2_scaled.ravel())

regressor_rna_saude.score(X_plano_saude2_scaled, y_plano_saude2_scaled)


grafico = px.scatter(x = X_plano_saude2_scaled.ravel(), y = y_plano_saude2_scaled.ravel())
grafico.add_scatter(x= X_plano_saude2_scaled.ravel(), y = regressor_rna_saude.predict(X_plano_saude2_scaled), name = "Regressão")
grafico.show()

novo = [[40]]
novo = scaler_X.transform(novo)
regressor_rna_saude.predict(novo)
#scaler_y.inverse_transform(regressor_rna_saude.predict(novo)) ERRO! ESTE CARALHO NÃO RODA NEM QUE A POHA! VSF!!!!

