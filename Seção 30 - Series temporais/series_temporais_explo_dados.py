# Etapa 2: Importação das bibliotecas"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima

## Etapa 3: Carregamento da base de dados"""
dataset = pd.read_csv("AirPassengers.csv")

dateparse = lambda dates: pd.datetime.strptime(dates, "%Y-%m") #Convertendo o formato dos dados para o Ano e Mês
dataset = pd.read_csv("AirPassengers.csv", parse_dates= ["Month"], index_col= "Month", date_parser= dateparse) #Colocando a coluna do mês como indice fazendo assim o dataset uma série temporal para manipular os dados

dataset.head()

time_series = dataset["#Passengers"]


# Etapa 4: Exploração da série temporal
time_series[1]
time_series["1949-02"]
time_series[datetime(1949,2,1)]
time_series["1950-01-01":"1950-07-31"]
time_series[:"1950-07-31"]
time_series["1950"]
time_series.index.max()
time_series.index.min()
plt.plot(time_series)

time_series_ano = time_series.resample("A").sum()
plt.plot(time_series_ano)

time_series_mes = time_series.groupby([lambda x: x.month]).sum()
plt.plot(time_series_mes)

time_series_datas = time_series["1960-01-01": "1960-12-01"]
plt.plot(time_series_datas)


#Etapa 5: Decomposição da série temporal
decomposicao = seasonal_decompose(time_series) #Fazer a divisão e analizar parte por parte a serie temporal
tendencia = decomposicao.trend #Tendencia da série temporal
sazonal = decomposicao.seasonal #Efeito sazonal
aleatorio = decomposicao.resid #Aleatorio

plt.plot(tendencia) #TENDENCIA DE CRESCIMENTO NO NUMERO DE VOOS POR MÊS

plt.plot(sazonal) #EFEITO SAZONAL

plt.plot(aleatorio) #ELEMENTO ALEATORIO, EVENTOS QUE ACONTECERAM EM DETERMINADOS TEMPOS//FENOMENOS QUE NÃO PODEM SER CONTROLADOS



#Etapa 6: Previsões com ARIMA
#Parâmetros P, Q e D
model = auto_arima(time_series, order=(2,1,2)) #Algoritmo vai uscar qual é o melhor conjunto desses parâmetros

predictions = model.predict(n_periods=12) #Previsões para doze meses para frente
#Previsão de passageiros por cada um dos meses de janeiro há dezembro


#Etapa 7: Gráfico das previsões
len(time_series)

train = time_series[:130]
train.shape

train.index.min(), train.index.max()

test = time_series[130:]
test.shape

test.index.min(), test.index.max()

model2 = auto_arima(train, suppress_warnings=True)

prediction = pd.DataFrame(model2.predict(n_periods=14), index=test.index)
prediction.columns = ["passengers_predictions"]
prediction

plt.figure(figsize=(8,5))
plt.plot(train, label = "Training")
plt.plot(test, label = "Test")
plt.plot(prediction, label = "Predictions")
plt.legend();