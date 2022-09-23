import pandas as pd
import string
from sklearn.metrics import accuracy_score
import spacy
import random
import seaborn as sns
import numpy as np 

#ETAPA 2: CARREGAMENTO DA BASE DE DADOS
base_dados = pd.read_csv("base_treinamento.txt", encoding="utf-8")
base_dados.shape

base_dados.head(10)

sns.countplot(base_dados["emocao"], label = "Contagem");

#ETAPA 3: FUNÇÃO PARA PRÉ-PROCESSAMENTO DOS TEXTOS
pontuacoes = string.punctuation

from spacy.lang.pt.stop_words import STOP_WORDS
stop_words = STOP_WORDS

pln = spacy.load("pt_core_news_sm")

def preprocessamento(texto):
    texto = texto.lower()
    documento = pln(texto)

    lista = []
    for token in documento:
        #lista.append(token.text)
        lista.append(token.lemma_)


    lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in pontuacoes]
    lista = " ".join([str(elemento) for elemento in lista if not elemento.isdigit()])
    return lista

teste = preprocessamento("Estou apendendo 1 20 13 processamento de linguagem natural, curso em Curitiba.")

#ETAPA 4: PRÉ-PROCESSAMENTO DA BASE DE DADOS
base_dados.head(10)
base_dados["texto"] = base_dados["texto"].apply(preprocessamento) #Aplicação da função em cada um dos textos
base_dados.head(10)

exemplor_base_dados = [["este trabalho é agradavel", {"ALEGRIA": True, "MEDO": False}, 
                        "este lugar continua assustador", {"ALEGRIA": False, "MEDO": True}]]

type(exemplor_base_dados) #Lista
type(exemplor_base_dados[0][1]) #Dicionário

base_dados_final = []
for texto, emocao in zip(base_dados["texto"], base_dados["emocao"]):
    #print(texto, emocao)
    if emocao == "alegria":
        dic = ({"ALEGRIA": True, "MEDO": False})
    elif emocao == "medo":
        dic = ({"ALEGRIA": False, "MEDO": True})
    base_dados_final.append([texto, dic.copy()])
len(base_dados_final)

base_dados_final[0]

#ETAPA 5: CRIAÇÃO DO CLASSIFICADOR 
modelo = spacy.blank("pt")
categorias = modelo.create_pipe("textcat")
categorias.add_label("ALEGRIA")
categorias.add_label("MEDO")
modelo.add_pipe("sentencizer")
historico = []

#Treinamento
modelo.begin_training() #------------------ATENÇÃO!!! CODIGO ESTÁ OBSOLETO
for epoca in range(1000):
    random.shuffle(base_dados_final)
    losses = {}
    for batch in spacy.util.minibatch(base_dados_final, 30):
        textos = [modelo(texto) for texto, entities in batch]
        annotations = [{"cats": entities} for texto, entities in batch]
        modelo.update(textos, annotations, losses= losses)
    if epoca % 100 == 0:
        print(losses)
        historico.append(losses)

historico_loss = []
for i in historico:
    historico_loss.append(i.get("textcat"))

historico_loss = np.array(historico_loss)

import matplotlib.pyplot as plt
plt.plot(historico_loss)
plt.title("Progessão do erro")
plt.xlabel("Épocas")
plt.ylabel("Erro")

modelo.to_disck("modelo")

#ETAPA 6: TESTES COM UMA FRASE
modelo_carregado = spacy.load("modelo")
texto_positivo = "eu adoro cor dos seus olhos"

texto_positivo = preprocessamento(texto_positivo)
previsao = modelo_carregado(texto_positivo)

previsao.cats

texto_negativo = "estou com medo dele"
previsao = modelo_carregado(preprocessamento(texto_negativo))
previsao.cats

#ETAPA 7: AVALIAÇÃO DO MODELO
# --- Avaliação na base de treinamento
previsoes = []
for texto in base_dados["texto"]:
    previsao = modelo_carregado(texto)
    previsao.append(previsao.cats)

previsoes_final = []
for previsao in previsoes:
    if previsao["ALEGRIA"] > previsao["MEDO"]:
        previsoes_final.append("alegria")
    else:
        previsoes_final.append("medo")
previsoes_final = np.array(previsoes_final)

respostas_reais = base_dados["emocao"].values


from sklearn.metrics import confusion_matrix, accuracy_score
accuracy_score(respostas_reais, previsoes_final)

cm = confusion_matrix(respostas_reais, previsoes_final)

# --- Avaliação na base de dados teste
base_dados_teste = pd.read_csv("base_teste.txt", encoding= "utf-8" )
base_dados_teste.head()

base_dados_teste["texto"] = base_dados_teste["texto"].apply(preprocessamento)
base_dados_teste.head()

previsoes = []
for texto in base_dados_teste["texto"]:
    previsao = modelo_carregado(texto)
    previsao.append(previsao.cats)

previsoes_final = []
for previsao in previsoes:
    if previsao["ALEGRIA"] > previsao["MEDO"]:
        previsoes_final.append("alegria")
    else:
        previsoes_final.append("medo")
previsoes_final = np.array(previsoes_final)

respostas_reais = base_dados_teste["emocao"].values

accuracy_score(respostas_reais, previsoes_final)
#0.57
cm = confusion_matrix(respostas_reais, previsoes_final)

#EXERCICIO! UTILIZAR A BASE DE DADOS DO TWITTER
# 099 DE PRECISÃO COM ESSA BASE DE DADOS!!