import pickle
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

with open("risco_credito.pkl", "rb") as f:
    X_risco_credito, y_risco_credito = pickle.load(f)

arvore_risco_credito = DecisionTreeClassifier(criterion="entropy")
arvore_risco_credito.fit(X_risco_credito, y_risco_credito)

arvore_risco_credito.feature_importances_ #Maiores ganhos de informações

from sklearn import tree
previsores = ["historia", "divida", "garantia", "renda"]
figura, eixos = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
tree.plot_tree(arvore_risco_credito, feature_names=previsores, 
               class_names=arvore_risco_credito.classes_, filled=True);

# historia boa (0), dívida alta (0), garantias nenhuma (1), renda > 35k (2)
#historia ruim (2), divida alta (0), garantias adequada (0), renda < 15k (0)
previsao = arvore_risco_credito.predict([[0,0,1,2], [2,0,0,0]])