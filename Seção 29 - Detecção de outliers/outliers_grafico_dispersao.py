import pandas as pd
import plotly.express as px

base_credit = pd.read_csv("credit_data.csv")

base_credit.isnull().sum()
base_credit.dropna(inplace=True)
base_credit.isnull().sum()

#Income x Age
grafico = px.scatter(x = base_credit["income"], y = base_credit["age"])
grafico.show()

#Income x loan
grafico = px.scatter(x = base_credit["income"], y = base_credit["loan"])
grafico.show()

#Age x loan
grafico = px.scatter(x = base_credit["age"], y = base_credit["loan"])
grafico.show()


base_census = pd.read_csv("census.csv")

#Age x final weight
grafico = px.scatter(x = base_census["age"], y = base_census["final.weight"])
grafico.show()
