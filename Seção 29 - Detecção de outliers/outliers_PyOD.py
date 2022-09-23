import pandas as pd
import plotly.express as px
import numpy as np


base_credit = pd.read_csv("credit_data.csv")
base_census = pd.read_csv("census.csv")
base_credit.dropna(inplace=True)


from pyod.models.knn import KNN #Essa biblioteca faz leva em consideração todos os atributos

detector = KNN()
detector.fit(base_credit.iloc[:,1:4])

previsoes = detector.labels_

np.unique(previsoes, return_counts=True) #1797 não são considerados outliers, 200 são.

confianca_previsoes = detector.decision_scores_ #Valores de distancia dos registros

outliers = []
for i in range(len(previsoes)):
    if previsoes[i] == 1:
        outliers.append(i)
print(outliers)

lista_outliers = base_credit.iloc[outliers,:] #ENVIAR O ARQUIVO DE OUTLIERS 

