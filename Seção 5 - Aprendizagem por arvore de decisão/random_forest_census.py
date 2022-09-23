import pickle #Biblioteca para abrir o arquivo pkl
from sklearn.metrics import accuracy_score, classification_report #Funções para analise dos resultados 
from sklearn.ensemble import RandomForestClassifier #Função do algoritmo de ML
from yellowbrick.classifier import ConfusionMatrix #Função para a matriz de confusão

#Abrindo o arquivo ja Pré-processado
with open("census.pkl", "rb") as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

#Analizando a estrutura das bases de treinamento e teste
X_census_treinamento.shape, y_census_treinamento.shape
X_census_teste.shape, y_census_teste.shape

random_forest_census = RandomForestClassifier(n_estimators=650, criterion="entropy", random_state=0)
random_forest_census.fit(X_census_treinamento, y_census_treinamento)

#Teste do algoritmo
previsoes = random_forest_census.predict(X_census_teste)
y_census_teste #Resposta do algoritmo

accuracy_score(y_census_teste, previsoes) 
#Precisão de 84,68% com arvores entre 500 e 650

cm = ConfusionMatrix(random_forest_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)

print(classification_report(y_census_teste, previsoes))