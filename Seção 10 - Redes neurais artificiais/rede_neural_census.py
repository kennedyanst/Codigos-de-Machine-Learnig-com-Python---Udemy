import pickle
from sklearn.neural_network import MLPClassifier


with open("../census.pkl", "rb") as f:
    X_census_treinamento, y_census_treinamento, X_census_teste, y_census_teste = pickle.load(f)

X_census_treinamento.shape, y_census_treinamento.shape
X_census_teste.shape, y_census_teste.shape


rede_neural_census = MLPClassifier(max_iter=1000, activation="relu", solver="adam", tol=0.000010, verbose=True, hidden_layer_sizes=(55,55)) #verose = olhar o progresso de aprendizagem
rede_neural_census.fit(X_census_treinamento, y_census_treinamento)

previsoes = rede_neural_census.predict(X_census_teste)
y_census_teste

from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_census_teste, previsoes) #PRECISÃO DE 81,08%

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(rede_neural_census)
cm.fit(X_census_treinamento, y_census_treinamento)
cm.score(X_census_teste, y_census_teste)

print(classification_report(y_census_teste, previsoes))