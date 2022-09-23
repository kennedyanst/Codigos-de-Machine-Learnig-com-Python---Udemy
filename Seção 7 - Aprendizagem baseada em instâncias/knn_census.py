import pickle #Biblioteca para abrir o arquivo pkl
from sklearn.metrics import accuracy_score, classification_report #Funções para analise dos resultados 
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.classifier import ConfusionMatrix #Função para a matriz de confusão

#Abrindo o arquivo ja Pré-processado
with open("census.pkl", "rb") as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

#Analizando a estrutura das bases de treinamento e teste
X_census_treinamento.shape, y_census_treinamento.shape
X_census_teste.shape, y_census_teste.shape

#IMPLEMENTANDO O KNN
knn_census = KNeighborsClassifier(n_neighbors=35)
knn_census.fit(X_census_treinamento, y_census_treinamento)

previsoes = knn_census.predict(X_census_teste)

accuracy_score(y_census_teste, previsoes) #Precisão de 82,56%

cm = ConfusionMatrix(knn_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)

print (classification_report(y_census_teste, previsoes))