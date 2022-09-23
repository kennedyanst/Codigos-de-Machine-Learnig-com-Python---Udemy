import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

base_census = pd.read_csv("census.csv")

X_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values


from sklearn.preprocessing import LabelEncoder
label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_ralationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

X_census[:,1] = label_encoder_workclass.fit_transform(X_census[:,1])
X_census[:,3] = label_encoder_education.fit_transform(X_census[:,3]) 
X_census[:,5] = label_encoder_marital.fit_transform(X_census[:,5]) 
X_census[:,6] = label_encoder_occupation.fit_transform(X_census[:,6]) 
X_census[:,7] = label_encoder_ralationship.fit_transform(X_census[:,7]) 
X_census[:,8] = label_encoder_race.fit_transform(X_census[:,8]) 
X_census[:,9] = label_encoder_sex.fit_transform(X_census[:,9]) 
X_census[:,13] = label_encoder_country.fit_transform(X_census[:,13]) 

from sklearn.preprocessing import StandardScaler
scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)

from sklearn.model_selection import train_test_split
X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = train_test_split(X_census, y_census, test_size = 0.15, random_state=0) #Random_state = 0: Faz sempre ter a mesma divisão
X_census_treinamento.shape, X_census_teste.shape


#MUITO PESADO PARA RODAR
# APLICANDO O KERNEL PCA PARA DIMINUIR A DIMENSIONALIDA
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=8, kernel="rbf")
X_census_treinamento_kpca = kpca.fit_transform(X_census_treinamento)
X_census_teste_kpca = kpca.transform(X_census_teste)


from sklearn.ensemble import RandomForestClassifier
random_forest_census_kpca = RandomForestClassifier(n_estimators=40, criterion = "entropy", random_state=0)
random_forest_census_kpca.fit(X_census_treinamento_kpca, y_census_treinamento)

previsoes = random_forest_census_kpca.predict(X_census_teste_kpca)

from sklearn.metrics import accuracy_score
accuracy_score(y_census_teste, previsoes) #0.823
