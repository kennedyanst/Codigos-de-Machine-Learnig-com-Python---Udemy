import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from yellowbrick.classifier import ConfusionMatrix

#Importando os arquivos 
with open("census.pkl", "rb") as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

#Analizando a estrutura das bases de treinamento e teste
X_census_treinamento.shape, y_census_treinamento.shape
X_census_teste.shape, y_census_teste.shape


naive_census = GaussianNB()
naive_census.fit(X_census_treinamento, y_census_treinamento) #Treinamento 
previsoes = naive_census.predict(X_census_teste) #Teste
previsoes

#Comparando a base de  yteste com os resultados da previsão
accuracy_score(y_census_teste, previsoes) #não executar o escalonamento o resultado fica melhor. 

#Analizando a precisão por uma matriz de confusão 
cm = ConfusionMatrix(naive_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)

print (classification_report(y_census_teste, previsoes))