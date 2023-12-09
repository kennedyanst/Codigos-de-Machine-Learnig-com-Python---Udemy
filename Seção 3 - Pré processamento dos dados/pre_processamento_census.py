import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#Impostação da base census
base_census = pd.read_csv("../census.csv")
base_census.describe() #Dados estatisticos
base_census.isnull().sum() #VERIFICAR SE EXISTE VALORES FALTANTES 

#Visualização dos dados
# np.unique(base_census["income"], return_counts=True) #Encontrando os atributos em cada uma das classes. 7508 ganha mais de 50k dols.
# sns.countplot(x = base_census["income"]);
# plt.hist(x = base_census["age"]);
# plt.hist(x= base_census["education.num"]); #Tempo de estudo
# plt.hist(x = base_census["hour.per.week"]); #Trabalho de horas por semana

# grafico = px.treemap(base_census, path = ["education", "income", "workclass", "age"]) #GRAFICO PARA AGRUPAMENTO DE DADOS!
# grafico.show()

# grafico = px.parallel_categories(base_census, dimensions = ["education", "income"]) #Grafico de categorias paralelas.
# grafico.show()


#Divisão entre previsores(X) e classe(Y)
base_census.columns
X_census = base_census.iloc[:, 0:14].values
y_census = base_census.iloc[:, 14].values


#Tratamento dos atributos categóricos - LabelEncoder
from sklearn.preprocessing import LabelEncoder

X_census[0]

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


#OneHotEncoder - Evita a diferença dos pesos das colunas
#len(np.unique(base_census["occupation"]))

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ondehotencoder_census = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder = "passthrough")
X_census = ondehotencoder_census.fit_transform(X_census).toarray()
X_census[0]
X_census.shape


#ESCALONAMENTO DOS VALORES
from sklearn.preprocessing import StandardScaler
scaler_census = StandardScaler()
X_census = scaler_census.fit_transform(X_census)


#BASE DE TREINO E BASE DE TESTE
from sklearn.model_selection import train_test_split
X_census_treinamento, X_census_teste, y_census_treinamento, y_census_teste = train_test_split(X_census, y_census, test_size = 0.15, random_state=0) #Random_state = 0: Faz sempre ter a mesma divisão
X_census_treinamento.shape
y_census_treinamento.shape
X_census_teste.shape, y_census_teste.shape


#SALVANDO AS VARIAVEIS DE TREINO E TESTE
import pickle
with open("census1.pkl", mode = "wb") as f:
    pickle.dump([X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste], f)