import bs4 as bs #Leitura e processamento de dados na web
import urllib.request
import spacy


dados = urllib.request.urlopen("https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial")
dados = dados.read() #Todo codigo HTML da pagina 
dados_html = bs.BeautifulSoup(dados, features="html.parser") #lxml: fazer leitura de arquivos HTML ou arquivos XMLM

paragrafos = dados_html.find_all("p") #Extraindo as tags de texto = </p>
len(paragrafos) #112 paragrafos
paragrafos[15]

paragrafos[8].text #Tirando os paragrafos e só retornando os textos

#Extraindo todo o conteudo da pagina
conteudo = ""
for p in paragrafos:
    conteudo += p.text

conteudo = conteudo.lower() #Tem que transformar em letras minusculas em linguagem natural, pq o algoritmo pode considerar Maius e minus como palavras diferentes 


#--------NUVEM DE PALAVRAS NO PYTHON
from spacy.lang.pt.stop_words import STOP_WORDS
pln = spacy.load("pt_core_news_sm")
#len(STOP_WORDS)

doc = pln(conteudo)
lista_token = []
for token in doc:
    lista_token.append(token.text)

print(lista_token)


sem_stop = []
for palavra in lista_token:
    if pln.vocab[palavra].is_stop == False:
        sem_stop.append(palavra)



from matplotlib.colors import ListedColormap
color_map = ListedColormap(["orange", "green", "red", "magenta"])

from wordcloud import WordCloud
cloud = WordCloud(background_color="white", max_words=100, colormap=color_map)

import matplotlib.pyplot as plt
cloud = cloud.generate(" ".join(sem_stop))
plt.figure(figsize=(15,15))
plt.imshow(cloud)
plt.axis("off")
plt.show()

