import pickle
from sklearn.neural_network import MLPClassifier


with open("credit.pkl", "rb") as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

X_credit_treinamento.shape, y_credit_treinamento.shape
X_credit_teste.shape, y_credit_teste.shape

rede_neural_credit = MLPClassifier(max_iter=3500, verbose=True, tol=0.000000100, solver= "adam", activation= "relu", hidden_layer_sizes=(20,20))
rede_neural_credit.fit(X_credit_treinamento, y_credit_treinamento) #Maximum interation = Epocas. 

previsoes = rede_neural_credit.predict(X_credit_teste)
y_credit_teste

from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_credit_teste, previsoes) #99,8% DE PRECISÃO. 

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(rede_neural_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)

print(classification_report(y_credit_teste, previsoes)) #Obs: Estou de pau duro com esse resultado...