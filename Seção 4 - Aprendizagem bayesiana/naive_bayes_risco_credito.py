import pandas as pd
from sklearn.naive_bayes import GaussianNB

#Selecionando a ase de dados
base_risco_credito = pd.read_csv("risco_credito.csv")

#Divindindo entre previsores e classe (X, y)
X_risco_credito =  base_risco_credito.iloc[:, 0:4].values
y_risco_credito = base_risco_credito.iloc[:, 4].values

#Fazendo o Encode dos dados previsores 
from sklearn.preprocessing import LabelEncoder
label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantia = LabelEncoder()
label_encoder_renda = LabelEncoder()

X_risco_credito[:,0] = label_encoder_historia.fit_transform(X_risco_credito[:,0])
X_risco_credito[:,1] = label_encoder_divida.fit_transform(X_risco_credito[:,1])
X_risco_credito[:,2] = label_encoder_garantia.fit_transform(X_risco_credito[:,2])
X_risco_credito[:,3] = label_encoder_renda.fit_transform(X_risco_credito[:,3])

#Salvando os dados 
import pickle
with open("risco_credito.pkl", "wb") as f:
    pickle.dump([X_risco_credito, y_risco_credito], f)

#Dividindo entre treinamento e teste
naive_risco_credito = GaussianNB()
naive_risco_credito.fit(X_risco_credito, y_risco_credito)

#Previsoes do NAIVE BAYES. 
# historia boa (0), dívida alta (0), garantias nenhuma (1), renda > 35k (2)
#historia ruim (2), divida alta (0), garantias adequada (0), renda < 15k (0)
previsao = naive_risco_credito.predict([[0,0,1,2], [2,0,0,0]])

naive_risco_credito.classes_
naive_risco_credito.class_count_
naive_risco_credito.class_prior_
