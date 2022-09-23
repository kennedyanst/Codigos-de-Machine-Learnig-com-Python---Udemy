import pickle
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


with open("credit.pkl", "rb") as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

X_credit_treinamento.shape, y_credit_treinamento.shape
X_credit_teste.shape, y_credit_teste.shape

arvore_credit = DecisionTreeClassifier(criterion="entropy", random_state = 0)
arvore_credit.fit(X_credit_treinamento, y_credit_treinamento)

previsoes = arvore_credit.predict(X_credit_teste)
y_credit_teste

from sklearn.metrics import accuracy_score, classification_report
accuracy_score(y_credit_teste, previsoes) #PRECISÃO DE 98,2%

from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(arvore_credit)
cm.fit(X_credit_treinamento, y_credit_treinamento)
cm.score(X_credit_teste, y_credit_teste) #Plot da imagem da matriz de confusão
print(classification_report(y_credit_teste, previsoes))


from sklearn import tree
previsores = ["income", "age", "loan"]
fig, eixos = plt.subplots(nrows=1, ncols=1, figsize=(20,20))
tree.plot_tree(arvore_credit, feature_names=previsores, 
               class_names= ["0", "1"], filled=True); #Grafico da arvore de decisão.
fig.savefig("arvore_credit.png") #Salvando a imagem da arvore em um arquivo png