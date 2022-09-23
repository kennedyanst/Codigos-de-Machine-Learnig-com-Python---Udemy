from sklearn.ensemble import RandomForestClassifier
import pickle


with open("credit.pkl", "rb") as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

X_credit_treinamento.shape, y_credit_treinamento.shape
X_credit_teste.shape, y_credit_teste.shape

random_forest_credit = RandomForestClassifier(n_estimators= 80, criterion="entropy", random_state=0)
random_forest_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = random_forest_credit.predict(X_credit_teste)
y_credit_teste

from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_credit_teste, previsoes)
#PRECISÃO DE 98,4% (n_estimators = 40: tb fica com a mesma precisão)

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(random_forest_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste)

print(classification_report(y_credit_teste, previsoes))