#Série temporal com Facebook Prophet - Previsão de visualização diaria de paginas da web
#RODAR NO GOOGLE COLAB

from matplotlib.pyplot import xlabel, ylabel
from fbprophet import Prophet
import pandas as pd

dataset = pd.read_csv("page_wikipedia.csv")
dataset.describe()
dataset.hist();

dataset = dataset[["date", "views"]].rename(columns = {"date": "ds", "views": "y"}) #Renomear para a padronização do Prophet

dataset = dataset.sort_values(by = "ds") #Datas ordenadas


#CONSTRUÇÃO DO MODELO
model = Prophet()
model.fit(dataset)

future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

forecast.head()

len(dataset), len(forecast)

len(forecast) - len(dataset)

forecast.tail(90) #VISUALIZANDO AS PREVISÕES


#GRAFICO DAS PREVISÕES
model.plot(forecast, xlabel = "Date", ylabel = "Views");
model.plot_components(forecast);
from fbprophet.plot import plot_plotly, plot_components_plotly
plot_plotly(model, forecast)
plot_components_plotly(model, forecast)