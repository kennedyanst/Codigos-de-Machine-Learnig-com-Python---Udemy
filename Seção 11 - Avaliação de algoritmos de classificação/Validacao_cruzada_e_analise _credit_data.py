from sklearn.model_selection import cross_val_score, KFold
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

    
resultados_arvore = []
resultados_random_forest = []
resultados_knn = []
resultados_regressao_logistica = []
resultados_svm = []
resultados_rede_neural =[]



#300 testes para cada um dos algoritmos
for i in range(30):
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)

    #Árvore de decisão
    arvore = DecisionTreeClassifier(criterion="entropy", min_samples_leaf=1, min_samples_split=5, splitter="best")
    score = cross_val_score(arvore, X_credit, y_credit, cv = kfold)
    print (score)
    print(score.mean())
    resultados_arvore.append(score.mean())

    #Random forest
    random_forest = RandomForestClassifier(criterion='entropy', min_samples_leaf= 1, min_samples_split= 5, n_estimators= 40)
    score = cross_val_score(random_forest, X_credit, y_credit, cv = kfold)
    print (score)
    print(score.mean())
    resultados_random_forest.append(score.mean())
    
    #KNN
    knn = KNeighborsClassifier(n_neighbors=5, metric = "minkowski", p=2)
    score = cross_val_score(knn, X_credit, y_credit, cv = kfold)
    print (score)
    print(score.mean())
    resultados_knn.append(score.mean())

    #Regressão logística
    regressao_logistica = LogisticRegression(C= 1.0, solver= 'lbfgs', tol= 0.0001)
    score = cross_val_score(regressao_logistica, X_credit, y_credit, cv = kfold)
    print (score)
    print(score.mean())
    resultados_regressao_logistica.append(score.mean())

    #SVM
    svm = SVC(C= 6.0, kernel= 'rbf', tol= 0.001)
    score = cross_val_score(svm, X_credit, y_credit, cv = kfold)
    print (score)
    print(score.mean())
    resultados_svm.append(score.mean())

    #Rede Neural
    rede_neural = MLPClassifier(solver= "adam", activation= "relu", batch_size=56)
    score = cross_val_score(rede_neural, X_credit, y_credit, cv = kfold)
    print (score)
    print(score.mean())
    resultados_rede_neural.append(score.mean())
    print(resultados_rede_neural)
    

# ------------ANÁLISE DOS RESULTADOS
import pandas as pd


resultados = pd.DataFrame({'Arvore': resultados_arvore, 'Random forest': resultados_random_forest, 
                           'KNN': resultados_knn, 'Regressao logistica': resultados_regressao_logistica,
                           'SVM': resultados_svm, 'Rede neural': resultados_rede_neural})

resultados.describe() #std = desvio padrão. Quanto menor, melhor!

#Coeficiente de váriação
(resultados.std() / resultados.mean()) * 100

#Respondendo qual é o melhor algoritmo, estatisticamente 
#Os dados precisam está em uma distribuição normal

alpha = 0.05 #Confiança de 95% do teste

#Teste de SHAPIRO 
from scipy.stats import shapiro
shapiro(resultados_arvore), shapiro(resultados_random_forest), shapiro(resultados_knn), shapiro(resultados_regressao_logistica), shapiro(resultados_svm), shapiro(resultados_rede_neural)
# Como o número de p (pvalue) é menor apenas nas redes neurais, em comparação ao número de alpha, esse algoritmo foi o unico a que rejeitou a hipotese nula e aeitou a hipotese alternativa. 
# Indicando que essa é uma distribuição não normal, fazendo assim possiel a plicação do teste de ANOVA e do teste de Tukey


#Analizando o grafico da normal
import seaborn as sns
sns.displot(resultados_arvore, kind= "kde");

sns.displot(resultados_random_forest, kind= "kde");

sns.displot(resultados_knn, kind= "kde");

sns.displot(resultados_regressao_logistica, kind= "kde");

sns.displot(resultados_svm, kind= "kde");

sns.displot(resultados_rede_neural, kind= "kde");


# ---- TESTE ANOVA E DE TUKEY (Só é possivel em dados que estejam em uma distribuição normal, usando o teste de shapiro)
from scipy.stats import f_oneway

_, p = f_oneway(resultados_arvore, resultados_random_forest, resultados_knn, resultados_regressao_logistica, resultados_svm, resultados_rede_neural)

alpha = 0.05
if p <= alpha:
    print("Hipotese nula rejeitada. Dados são diferentes")
else:
    print("Hipótese alternativa rejeitada. Resultados são iguais")
#RESPOSTA: Hipotese nula rejeitada. Dados são diferentes. 
#Significa que os resultados da arvore é diferente do random forest, que é diferente do knn e assim por diante. 
#Chegando a conclusão que os dados são diferentes, agora pode aplicar o teste de Tukey, para saber qual é o melhor algoritmo. 

resultados_algoritmos = {"Accuracy": np.concatenate([resultados_arvore, resultados_random_forest, resultados_knn, resultados_regressao_logistica, resultados_svm, resultados_rede_neural]),
                          "Algoritmo": ['Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão','Árvore de decisão',
                                        'Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest','Random forest',
                                        'KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN','KNN',
                                        'Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística','Regressão logística',
                                        'SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM','SVM',
                                        'Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural','Rede neural']}

resultados_df = pd.DataFrame(resultados_algoritmos)
resultados_df.head(30).mean()
resultados_df.tail(30).mean()

#APLICAÇÃO DO TESTE DE TUKEY
from statsmodels.stats.multicomp import MultiComparison


compara_algoritmos = MultiComparison(resultados_df["Accuracy"], resultados_df["Algoritmo"])
teste_estatistico = compara_algoritmos.tukeyhsd()
print (teste_estatistico) #True =  Algoritmos possuem diferenças estatisticas significativas
                          #False = Algoritmos não são estatisticamente diferentes. 
resultados.mean()

teste_estatistico.plot_simultaneous(); #Grafico mostrando a superioridade dos algoritmos. REDE NEURAL melhor que todos os outros! 