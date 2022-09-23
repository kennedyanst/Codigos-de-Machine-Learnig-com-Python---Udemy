import pickle
import numpy as np


with open("credit.pkl", "rb") as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, y_credit_teste = pickle.load(f)

X_credit = np.concatenate((X_credit_treinamento, X_credit_teste), axis = 0)
y_credit = np.concatenate((y_credit_treinamento, y_credit_teste), axis = 0)


rede_neural = pickle.load(open("rede_neural_finalizado.sav", "rb"))
arvore = pickle.load(open("arvore_decisao_finalizado.sav", "rb"))
svm = pickle.load(open("svm_finalizado.sav", "rb"))

#Testando os algoritmos carregados. 
novo_registro = X_credit[0] #Paga = 0
novo_registro = X_credit[1999] #Não paga = 1
novo_registro = novo_registro.reshape(1,-1) #Colocando em formato de matriz
novo_registro.shape

rede_neural.predict(novo_registro)
arvore.predict(novo_registro)
svm.predict(novo_registro)