#----- ATENÇÃO: Rodar no Google colab...

import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import numpy as np


base_cartao = pd.read_csv("credit_card_clients.csv")
base_cartao["BILL TOTAL"] = base_cartao["BILL_AMT1"] + base_cartao["BILL_AMT2"] + base_cartao["BILL_AMT3"] + base_cartao["BILL_AMT4"] + base_cartao["BILL_AMT5"] + base_cartao["BILL_AMT6"]

X_cartao = base_cartao.iloc[:, [1, 25]].values
scaler_cartao = StandardScaler()
X_cartao = scaler_cartao.fit_transform(X_cartao)



import matplotlib.pyplot as plt 
from scipy.cluster.hierarchy import dendrogram, linkage

dendrograma = dendrogram(linkage(X_cartao, method = "ward"))
plt.title("Dendrograma")
plt.xlabel("Pessoas")
plt.ylabel("Distância")


from sklearn.cluster import AgglomerativeClustering

hc_cartao = AgglomerativeClustering(n_clusters=3, linkage = "ward", affinity="euclidean")
rotulos = hc_cartao.fit_predict(X_cartao)

grafico = px.scatter(x = X_cartao[:, 0], y = X_cartao[:, 1], color = rotulos)
grafico.show()