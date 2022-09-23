import pickle
from sklearn.naive_bayes import GaussianNB


with open("credit.pkl", "rb") as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

X_credit_treinamento.shape, y_credit_treinamento.shape
X_credit_teste.shape, y_credit_teste.shape

naive_credit_data = GaussianNB()
naive_credit_data.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = naive_credit_data.predict(X_credit_teste)
y_credit_teste

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy_score(y_credit_teste, previsoes)
#ACERTO DE 93,8%
confusion_matrix(y_credit_teste, previsoes)
#array([[428,   8],
#       [ 23,  41]]

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(naive_credit_data)
cm.fit(X_credit_treinamento, y_credit_teste)
cm.score(X_credit_teste, y_credit_teste)

print(classification_report(y_credit_teste, previsoes)) #recall = Precisão do algoritmo em identificar a classe correta.
