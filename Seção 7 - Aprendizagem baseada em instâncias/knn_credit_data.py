import pickle
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

with open("credit.pkl", "rb") as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

X_credit_treinamento.shape, y_credit_treinamento.shape
X_credit_teste.shape, y_credit_teste.shape

knn_credit = KNeighborsClassifier(n_neighbors=5, metric = "minkowski", p=2)
knn_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = knn_credit.predict(X_credit_teste)

from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_credit_teste, previsoes) #98,6% de PRECISÃO com padronização!

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(knn_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)

print(classification_report(y_credit_teste, previsoes))
