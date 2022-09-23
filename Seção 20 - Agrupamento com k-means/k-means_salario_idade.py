import plotly.express as px
import plotly.graph_objects as go
import numpy as np 
from sklearn.preprocessing import StandardScaler 


from sklearn.cluster import KMeans

X = [20, 27, 21, 37, 46, 53, 55, 47, 52, 32, 39, 41, 39, 48, 48]
Y = [1000, 1200, 2900, 1850, 900, 950, 2000, 2100, 3000, 5900, 4100, 5100, 7000, 5000, 6500]

grafico = px.scatter(x = X, y = Y)
grafico.show()

base_salario = np.array([[20, 1000], [27, 1200], [21, 2900], [37, 1850], [46, 900], [53, 950], 
                        [55, 2000], [47, 2100], [52, 3000], [32, 5900], [39, 4100], [41, 5100], 
                        [39, 7000], [48, 5000], [48, 6500]])#Estão em formato de vetor, precisa transformar em formato de matriz

scaler_salario = StandardScaler()
base_salario = scaler_salario.fit_transform(base_salario)

kmeans_salario = KMeans(n_clusters = 3)
kmeans_salario.fit(base_salario)

centroides = kmeans_salario.cluster_centers_

scaler_salario.inverse_transform(kmeans_salario.cluster_centers_)

rotulos = kmeans_salario.labels_

grafico1 = px.scatter(x = base_salario[:, 0], y = base_salario[:, 1], color = rotulos)
grafico2 = px.scatter(x = centroides[:, 0], y = centroides[:, 1], size = [12, 12, 12])
grafico3 = go.Figure(data = grafico1.data + grafico2.data)
grafico3.show()
