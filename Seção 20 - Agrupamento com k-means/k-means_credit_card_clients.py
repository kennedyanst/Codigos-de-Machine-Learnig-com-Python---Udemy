import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import numpy as np


base_cartao = pd.read_csv("credit_card_clients.csv")
base_cartao["BILL TOTAL"] = base_cartao["BILL_AMT1"] + base_cartao["BILL_AMT2"] + base_cartao["BILL_AMT3"] + base_cartao["BILL_AMT4"] + base_cartao["BILL_AMT5"] + base_cartao["BILL_AMT6"]

X_cartao = base_cartao.iloc[:, [1, 25]].values
scaler_cartao = StandardScaler()
X_cartao = scaler_cartao.fit_transform(X_cartao)

#---- WCSS: Metodo do cotovelo = Descobrir o número de clusters

wcss = []
for i in range(1, 11):
    kmeans_cartao = KMeans(n_clusters=i, random_state=0)
    kmeans_cartao.fit(X_cartao)
    wcss.append(kmeans_cartao.inertia_)
wcss

grafico = px.line(x = range(1,11), y = wcss)
grafico.show() #FAZER O AGRUPAMENTO COM 4 OU 5 CLUSTERS

kmeans_cartao = KMeans(n_clusters=4, random_state=0)
rotulos = kmeans_cartao.fit_predict(X_cartao)

grafico = px.scatter(x = X_cartao[:,0], y = X_cartao[:,1], color = rotulos)
grafico.show()

lista_clientes = np.column_stack((base_cartao, rotulos)) #Lista dos clientes por grupo
lista_clientes = lista_clientes[lista_clientes[:, 26]. argsort()]


#------------- MAIS ATRIBUTOS -----------
base_cartao.columns

X_cartao_mais = base_cartao.iloc[:, [1,2,3,4,5,25]]
scaler_cartao_mais = StandardScaler()
X_cartao_mais = scaler_cartao.fit_transform(X_cartao_mais)

wcss = []
for i in range(1, 11):
    kmeans_cartao_mais = KMeans(n_clusters=i, random_state=0)
    kmeans_cartao_mais.fit(X_cartao_mais)
    wcss.append(kmeans_cartao_mais.inertia_)
wcss


grafico = px.line(x = range(1,11), y = wcss)
grafico.show() #FAZER O AGRUPAMENTO COM 4 OU 5 CLUSTERS


kmeans_cartao_mais = KMeans(n_clusters=2, random_state=0)
rotulos = kmeans_cartao_mais.fit_predict(X_cartao_mais)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_cartao_mais_pca = pca.fit_transform(X_cartao_mais)
X_cartao_mais_pca.shape

grafico = px.scatter(x = X_cartao_mais_pca[:,0], y = X_cartao_mais_pca[:,1], color = rotulos)
grafico.show()


lista_clientes = np.column_stack((base_cartao, rotulos)) #Lista dos clientes por grupo
lista_clientes = lista_clientes[lista_clientes[:, 26]. argsort()]