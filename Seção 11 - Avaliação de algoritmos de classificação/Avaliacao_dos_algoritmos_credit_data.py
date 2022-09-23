# ----------AVALIAÇÃO DOS ALGORITMOS: CREDIT DATA
# - Naive Bayes:                93.80
# - Árvores de decisão:         98.20
# - Random forest:              98.40
# - Regras:                     97.40
# - Knn:                        98.60
# - Regressão logística:        94.60
# - SVM:                        98.80
# - Redes neurais:              99.60

# ----------Turning dos parâmetros com GridSearch

#Preparação dos dados
from sklearn.model_selection import GridSearchCV #Vai escolher os melhores paramentros para cada um dos algoritmos, pesquisando em grade. CV = Cross validation (Aplicação da validação cruzada)
from sklearn.tree import DecisionTreeClassifier #Árvore de decisão
from sklearn.ensemble import RandomForestClassifier #Random forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.linear_model import LogisticRegression #Regressão logística
from sklearn.svm import SVC #SVM
from sklearn.neural_network import MLPClassifier #Redes neurais
#(O naive bayes não tem paramentros relevantes para a aplicação do GridSearchCV. O regras utiliza o Orange e não teve resultados relevantes)

import pickle
with open("credit.pkl", "rb") as f:
    X_credit_treinamento, y_credit_treinamento, X_credit_teste, _y_credit_teste = pickle.load(f)

X_credit_treinamento.shape, y_credit_treinamento.shape
X_credit_teste.shape, _y_credit_teste.shape

import numpy as np
X_credit = np.concatenate((X_credit_treinamento, X_credit_teste), axis = 0) #Concatenação das linhas do X
y_credit = np.concatenate((y_credit_treinamento, _y_credit_teste), axis = 0) #Concatenação das linhas do y


# Árvore de decisão
paramentros = {"criterion": ["gini", "entropy"],
               "splitter": ["best", "random"],
               "min_samples_split": [2,5,10],
               "min_samples_leaf": [1,5,10]}

grid_search = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid=paramentros)
grid_search.fit(X_credit, y_credit)
melhores_paramentros = grid_search.best_params_
melhores_resultados = grid_search.best_score_
print(melhores_paramentros)
print(melhores_resultados)
#RESULTADOS:
#{'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'best'}
#0.983 - Aumento de 0,10%


# - Random forest:              98.40
paramentros = {"criterion": ["gini", "entropy"],
               "n_estimators": [10,40,100,150,200],
               "min_samples_split": [2,5,10,15,20],
               "min_samples_leaf": [1,5,10,15,20]}

grid_search = GridSearchCV(estimator = RandomForestClassifier(), param_grid=paramentros)
grid_search.fit(X_credit, y_credit)
melhores_paramentros = grid_search.best_params_
melhores_resultados = grid_search.best_score_
print(melhores_paramentros)
print(melhores_resultados)
#RESULTADOS:
#{'criterion': 'entropy', 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 40}
#0.9884999999999999


# - Knn:                        98.60
paramentros = {"n_neighbors": [3,5,10,20,25,50,100],
              "p": [1,2]}

grid_search = GridSearchCV(estimator = KNeighborsClassifier(), param_grid=paramentros)
grid_search.fit(X_credit, y_credit)
melhores_paramentros = grid_search.best_params_
melhores_resultados = grid_search.best_score_
print(melhores_paramentros)
print(melhores_resultados)
#RESULTADOS:
#{'n_neighbors': 25, 'p': 1}
#0.9804999999999999


# - Regressão logística:        94.60
paramentros = {"tol": [0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001],
               "C": [1.0, 1.5, 2.0, 3.0, 5.0],
               "solver": ["lbfgs", "sag", "saga"]}

grid_search = GridSearchCV(estimator = LogisticRegression(), param_grid=paramentros)
grid_search.fit(X_credit, y_credit)
melhores_paramentros = grid_search.best_params_
melhores_resultados = grid_search.best_score_
print(melhores_paramentros)
print(melhores_resultados)
#RESULTADOS:
#{'C': 1.0, 'solver': 'lbfgs', 'tol': 0.0001}
#0.9484999999999999


# - SVM:                        98.80
paramentros = {"tol": [0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
            "C": [1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 6.0],
            "kernel": ["rbf", "linear", "poly", "sigmoid"]}

grid_search = GridSearchCV(estimator = SVC(), param_grid=paramentros)
grid_search.fit(X_credit, y_credit)
melhores_paramentros = grid_search.best_params_
melhores_resultados = grid_search.best_score_
print(melhores_paramentros)
print(melhores_resultados)
#RESULTADOS:
#{'C': 6.0, 'kernel': 'rbf', 'tol': 0.001}
#0.9855


# - Redes neurais:              99.60
paramentros = {"activation": ["relu", "logistic", "tahn"],
               "solver": ["adam", "sgd"],
               "batch_size": [10, 56, 100, 200, 350]}

grid_search = GridSearchCV(estimator = MLPClassifier(), param_grid=paramentros)
grid_search.fit(X_credit, y_credit)
melhores_paramentros = grid_search.best_params_
melhores_resultados = grid_search.best_score_
print(melhores_paramentros)
print(melhores_resultados)
#RESULTADOS:
#{'activation': 'relu', 'batch_size': 56, 'solver': 'adam'}
#0.9964999999999999