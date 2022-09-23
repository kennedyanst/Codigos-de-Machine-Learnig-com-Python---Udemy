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



#------------------- BUSCA EM TEXO BUSCANDO O spacy
pln = spacy.load("")