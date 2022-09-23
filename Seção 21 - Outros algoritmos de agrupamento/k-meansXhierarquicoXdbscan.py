from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import numpy as np
import plotly.express as px


from sklearn import datasets

X_random, y_random = datasets.make_moons(n_samples=1500, noise = 0.03)

np.unique(y_random)

grafico = px.scatter(x = X_random[:,0], y = X_random[:,1])
grafico.show()


#APLICANDO O K-MEANS
kmeans = KMeans(n_clusters =2)
rotulos = kmeans.fit_predict(X_random)
grafico = px.scatter(x = X_random[:,0], y = X_random[:,1], color = rotulos)
grafico.show()
#Não é possivel usar o kmenas pq ele usa a distancia. 


#APLICANDO O AGRUPAMENTO HIERARQUICO 
hc = AgglomerativeClustering(n_clusters = 2, affinity= "euclidean", linkage = "ward")
rotulos = hc.fit_predict(X_random)
grafico = px.scatter(x = X_random[:,0], y = X_random[:,1], color = rotulos)
grafico.show()


#APLICANDO O DBSCAN
dbscan = DBSCAN(eps=0.1)
rotulos = dbscan.fit_predict(X_random)
grafico = px.scatter(x = X_random[:,0], y = X_random[:,1], color = rotulos)
grafico.show()
#MELHOR PARA ESSA BASE DE DADOS. 