import pickle
import numpy as np


with open("credit.pkl", "rb") as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

X_credit = np.concatenate((X_credit_treinamento, X_credit_teste), axis = 0)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis = 0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import  MLPClassifier
from sklearn.svm import SVC

classificador_rede_neural = MLPClassifier(activation = "relu", batch_size=56, solver="adam")
classificador_rede_neural.fit(X_credit, y_credit)

classificador_arvore = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=1, min_samples_split=5, splitter="best")
classificador_arvore.fit(X_credit, y_credit)

classificador_svm = SVC(C= 6.0, kernel= 'rbf', tol= 0.001)
classificador_svm.fit(X_credit, y_credit)


#SALVANDO OS CLASSIFICADORES EM DISCO.
pickle.dump(classificador_rede_neural, open("rede_neural_finalizado.sav", "wb"))
pickle.dump(classificador_arvore, open("arvore_decisao_finalizado.sav", "wb"))
pickle.dump(classificador_svm, open("svm_finalizado.sav", "wb"))
#Os arquivos .sav que será enviado ao desenvolvedor para ele fazer a interface grafica/web/desktop para ser usado. 

