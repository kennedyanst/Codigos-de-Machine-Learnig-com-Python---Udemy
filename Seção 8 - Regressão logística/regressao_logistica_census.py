import pickle
from sklearn.linear_model import LogisticRegression #Biblioteca para abrir o arquivo pkl
from sklearn.metrics import accuracy_score, classification_report #Funções para analise dos resultados 
from yellowbrick.classifier import ConfusionMatrix #Função para a matriz de confusão

#Abrindo o arquivo ja Pré-processado
with open("../census.pkl", "rb") as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

X_census_treinamento.shape, y_census_treinamento.shape
X_census_teste.shape, y_census_teste.shape

logistic_census = LogisticRegression(random_state= 1)
logistic_census.fit(X_census_treinamento, y_census_treinamento)

previsoes = logistic_census.predict(X_census_teste)
y_census_teste

accuracy_score(y_census_teste, previsoes) #84,33%

cm = ConfusionMatrix(logistic_census)
cm.fit(X_census_treinamento, y_census_teste)
cm.score(X_census_teste, y_census_teste)

print(classification_report(y_census_teste, previsoes))