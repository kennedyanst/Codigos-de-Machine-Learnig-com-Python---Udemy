import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

#ANALISANDO OS DADOS. 
base_credit = pd.read_csv("credit_data.csv") #defaulted = não pagou
#base_credit.head(10) #Cabeçalho dos (10) primeiros
#base_credit.tail(10) #Cabeçalho dos (10) ultimos 
#base_credit.describe() #Comando do pandas para descrever a base. std = desvio padrão

#base_credit[base_credit["income"] >= 69995.685578] #Maior renda!
#base_credit[base_credit["loan"] <= 1.377630] #Menor divida!


#VISUALIZAÇÃO DOS GRAFICOS
#np.unique(base_credit["default"], return_counts= True) #Contador dos valores unicos da coluna. 283 clientes não pagam o emprestimo. 
#sns.countplot(x = base_credit["default"]);
#plt.hist(x = base_credit["age"]);
#plt.hist(x = base_credit["income"]);
#plt.hist(x = base_credit["loan"]);
#GRAFICO DE DISPERÇÃO
#grafico = px.scatter_matrix(base_credit, dimensions = ["age", "income", "loan"], color = "default") 
#grafico.show()


#TRATAMENTO DE VALORES INCONSISTENTES
#base_credit.loc[base_credit["age"] < 0] #Comando loc é para encontrar
#base_credit[base_credit["age"] <0 ] #Sem o uso do comando loc 
# - Apagar a coluna inteira (de todos os registros da base de dados)
#base_credit2 = base_credit.drop("age", axis = 1)
# - Apagar somente os registros com valores inconsistentes
#base_credi3 = base_credit.drop(base_credit[base_credit["age"] < 0].index)
# - Preencher os valores inconsistente manualmente com as médias das idades
base_credit.mean()
base_credit["age"].mean()
base_credit["age"][base_credit["age"] > 0].mean() #Média das idades que não são negativas 40.92770044906149
base_credit.loc[base_credit["age"] < 0, "age" ] = 40.92 #Colocar o "age" se não ele vai mudar toda a linha para 40.92
base_credit.head(27)


#TRATAMENTO DE VALORES FALTANTES
base_credit.isnull() #True = valores faltantes
base_credit.isnull().sum() #3 registros no age que não foram preenchidos
base_credit.loc[pd.isnull(base_credit["age"])]

base_credit["age"].fillna(base_credit["age"].mean(), inplace = True) #Inplace quer dizer que vai alterar a variavel base_credit
base_credit.loc[(base_credit["clientid"] == 29) | (base_credit["clientid"] == 31) | (base_credit["clientid"] == 32)] #Verificando se os valores NaN foram mudados
base_credit.loc[(base_credit["clientid"].isin([29,31,32]))] #Verificando se os valores NaN foram mudados de uma forma mais simples


#DIVISÃO ENTRE PREVISOES E CLASSE
X_credit = base_credit.iloc[:, 1:4].values #Vai buscar todas a linhas das colunas income, age e loan. O .values é para deixar no formato do pandas 
type(X_credit)
y_credit =  base_credit.iloc[:, 4].values


#ESCALONAMENTO DE ATRIBUTOS - A diferença de escala entre os atributos, vai influenciar o resultado do algoritmo. O ideal é padronizar os valores deixando na mesma escala
# X_credit[:,0].min() #Menor renda
# X_credit[:,0].max() #Maior renda
# X_credit[:, 1].min() #Menor idade
# X_credit[:, 1].max() #Maior idade
# X_credit[:, 2].min() #Menor divida
# X_credit[:, 2].max() #Maior divida
#A função de padronização está no sklearn 
from sklearn.preprocessing import StandardScaler
scaler_credit = StandardScaler()
X_credit = scaler_credit.fit_transform(X_credit) #Todos os valores estão na mesma escala agora. 


#BASE DE TREINO E BASE DE TESTE
from sklearn.model_selection import train_test_split

X_credit_treinamento, X_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(X_credit, y_credit, test_size = 0.25, random_state=0)#Random_state = 0: Faz sempre ter a mesma divisão
X_credit_treinamento.shape
y_credit_treinamento.shape
X_credit_teste.shape, y_credit_teste.shape


#SALVANDO AS VARIAVEIS DE TREINO E TESTE
import pickle
with open("credit.pkl", mode = "wb") as f:
    pickle.dump([X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste], f)