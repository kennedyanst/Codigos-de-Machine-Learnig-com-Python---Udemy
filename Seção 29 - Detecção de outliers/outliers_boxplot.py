import pandas as pd
import plotly.express as px

base_credit = pd.read_csv("credit_data.csv")

base_credit.isnull().sum()
base_credit.dropna(inplace=True)
base_credit.isnull().sum()

#Outliers idade
grafico = px.box(base_credit, y = "age")
grafico.show()
#REGISTROS QUE FOGEM DO PADRÃO
outliers_age = base_credit[base_credit["age"] < 0]

#Outliers Divida
grafico = px.box(base_credit, y = "loan")
grafico.show()
#REGISTROS QUE FOGEM DO PADRÃO
outliers_loan = base_credit[base_credit["loan"] > 13300]
