import pickle #Biblioteca para abrir o arquivo pkl
from sklearn.metrics import accuracy_score, classification_report #Funções para analise dos resultados 
from sklearn.tree import DecisionTreeClassifier #Função do algoritmo de ML
from yellowbrick.classifier import ConfusionMatrix #Função para a matriz de confusão

#Abrindo o arquivo ja Pré-processado
with open("census.pkl", "rb") as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

#Analizando a estrutura das bases de treinamento e teste
X_census_treinamento.shape, y_census_treinamento.shape
X_census_teste.shape, y_census_teste.shape

#Criando a arvore de decisão
arvore_census = DecisionTreeClassifier(criterion="entropy", random_state=0)
arvore_census.fit(X_census_treinamento, y_census_treinamento) #Treinando o modelo

#Testando o algoritmo
previsoes = arvore_census.predict(X_census_teste)
accuracy_score(y_census_teste, previsoes) #Precisão de 80,8%

#Analizando a precisão por uma matriz de confusão 
cm = ConfusionMatrix(arvore_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)

#Analizando melhor a precisão em cada atributo da classe
print (classification_report(y_census_teste, previsoes))