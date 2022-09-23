from sklearn import svm
from sklearn.svm import SVC
import pickle
from sklearn.metrics import accuracy_score, classification_report
from yellowbrick.classifier import ConfusionMatrix

with open("credit.pkl", "rb") as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

X_credit_treinamento.shape, y_credit_treinamento.shape
X_credit_teste.shape, y_credit_teste.shape

#LINEAR
svm_credit = SVC(kernel="linear", random_state=1, C = 1.0)
svm_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = svm_credit.predict(X_credit_teste)
y_credit_teste

accuracy_score(y_credit_teste, previsoes) #94.6% de precisão


#POLINOMIAL
svm_credit = SVC(kernel="poly", random_state=1, C = 1.0)
svm_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = svm_credit.predict(X_credit_teste)
y_credit_teste

accuracy_score(y_credit_teste, previsoes) #96,8% de precisão


#SIGMOID
svm_credit = SVC(kernel="sigmoid", random_state=1, C = 1.0)
svm_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = svm_credit.predict(X_credit_teste)
y_credit_teste

accuracy_score(y_credit_teste, previsoes) #83,8% de precisão


#RBF
svm_credit = SVC(kernel="rbf", random_state=1, C = 1.0)
svm_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = svm_credit.predict(X_credit_teste)
y_credit_teste

accuracy_score(y_credit_teste, previsoes) #98,2% DE PRECISÃO


#RBF com o C = 2.0 (MELHOR RESULTADO COM ESSA BASE DE DADOS)
svm_credit = SVC(kernel="rbf", random_state=1, C = 2.0)
svm_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = svm_credit.predict(X_credit_teste)
y_credit_teste

accuracy_score(y_credit_teste, previsoes) #98,8% DE PRECISÃO


cm = ConfusionMatrix(svm_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)

print(classification_report(y_credit_teste, previsoes))