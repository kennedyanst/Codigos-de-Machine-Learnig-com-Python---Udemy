import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import numpy as np


base_cartao = pd.read_csv("credit_card_clients.csv")
base_cartao["BILL TOTAL"] = base_cartao["BILL_AMT1"] + base_cartao["BILL_AMT2"] + base_cartao["BILL_AMT3"] + base_cartao["BILL_AMT4"] + base_cartao["BILL_AMT5"] + base_cartao["BILL_AMT6"]

X_cartao = base_cartao.iloc[:, [1, 25]].values
scaler_cartao = StandardScaler()
X_cartao = scaler_cartao.fit_transform(X_cartao)


from sklearn.cluster import DBSCAN
dbscan_salario = DBSCAN(eps = 0.37, min_samples=5)
rotulos = dbscan_salario.fit_predict(X_cartao)


np.unique(rotulos, return_counts=True)


grafico = px.scatter(x = X_cartao[:,0], y = X_cartao[:,1], color = rotulos)
grafico.show()