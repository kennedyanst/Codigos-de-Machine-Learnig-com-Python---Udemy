import pandas as pd
import plotly.express as px
import numpy as np


base_plano_saude2 = pd.read_csv("plano_saude2.csv")

X_plano_saude2 = base_plano_saude2.iloc[:, 0:1].values #idade
y_plano_saude2 = base_plano_saude2.iloc[:, 1].values #custo do plano de saúde


# ------IMPORTANTE!!!!!------ PARA O SVM, PRECISA NORMALIZAR OS DADOS!!! Os outros algoritmos de regressão aplicam a normalização internamente
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_plano_saude2_scaled = scaler_X.fit_transform(X_plano_saude2)
scaler_y = StandardScaler()
y_plano_saude2_scaled = scaler_X.fit_transform(y_plano_saude2.reshape(-1,1))



# Kernel linear (Equivale a usar a regressão linear)
from sklearn.svm import SVR
regressor_svr_saude_linear = SVR(kernel = "linear")
regressor_svr_saude_linear.fit(X_plano_saude2, y_plano_saude2)

grafico = px.scatter(x = X_plano_saude2.ravel(), y = y_plano_saude2)
grafico.add_scatter(x= X_plano_saude2.ravel(), y = regressor_svr_saude_linear.predict(X_plano_saude2), name = "Regressão")
grafico.show()


#Kernel polinomial (Equivale a usar a regressão polinomial)
regressor_svr_saude_poly = SVR(kernel = "poly", degree = 4)
regressor_svr_saude_poly.fit(X_plano_saude2, y_plano_saude2)

grafico = px.scatter(x = X_plano_saude2.ravel(), y = y_plano_saude2)
grafico.add_scatter(x= X_plano_saude2.ravel(), y = regressor_svr_saude_poly.predict(X_plano_saude2), name = "Regressão")
grafico.show()


#Kernel rbf
regressor_svr_saude_rbf = SVR(kernel = "rbf")
regressor_svr_saude_rbf.fit(X_plano_saude2_scaled, y_plano_saude2_scaled.ravel())

grafico = px.scatter(x = X_plano_saude2_scaled.ravel(), y = y_plano_saude2_scaled.ravel())
grafico.add_scatter(x= X_plano_saude2_scaled.ravel(), y = regressor_svr_saude_rbf.predict(X_plano_saude2_scaled), name = "Regressão")
grafico.show()


novo = [[40]]
novo = scaler_X.transform(novo)



regressor_svr_saude_rbf.predict([[40]])
regressor_svr_saude_rbf.predict(novo)
#scaler_y.inverse_transform(regressor_svr_saude_rbf.predict(novo)) #----- ERRO FILHO DA PUTA QUE NÃO SAI!!!!!
