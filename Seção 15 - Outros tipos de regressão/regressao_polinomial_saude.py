import pandas as pd
import plotly.express as px

base_plano_saude2 = pd.read_csv("plano_saude2.csv")

X_plano_saude2 = base_plano_saude2.iloc[:, 0:1].values
y_plano_saude2 = base_plano_saude2.iloc[:, 1].values


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4) #4 vezes o atributo (n, n², n³, n na quarta)
X_plano_saude2_poly = poly.fit_transform(X_plano_saude2)


X_plano_saude2_poly.shape
X_plano_saude2_poly[0]

from sklearn.linear_model import LinearRegression
regressor_saude_polinomial = LinearRegression()
regressor_saude_polinomial.fit(X_plano_saude2_poly, y_plano_saude2)

regressor_saude_polinomial.intercept_
regressor_saude_polinomial.coef_

novo = [[40]]
novo = poly.transform(novo)

regressor_saude_polinomial.predict(novo) #O custo do plano de saude para uma pessoa de 40 anos, nesse algoritmo é de 1335.33 reais. 

previsoes = regressor_saude_polinomial.predict(X_plano_saude2_poly)

grafico = px.scatter(x = X_plano_saude2[:,0], y = y_plano_saude2)
grafico.add_scatter(x = X_plano_saude2[:,0], y = previsoes, name = "Regressão Polinomial")
grafico.show()