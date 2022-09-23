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


# -------- Buscas em texto com spaCy
pln = spacy.load("pt_core_news_sm")
string = "turing"
token_pesquisa = pln(string)

from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(pln.vocab) #pln.vocab tem todo o vocabulario da lingua portuguesa 
matcher.add("SEARCH", None, token_pesquisa)

doc = pln(conteudo)
matches = matcher(doc)
len(matches) #Encontrou a palavra "Turing" 12 vezes. 

doc[3035:3036] 
doc[3035-5:3036+5] #Vai buscar os 5 caracteres anteriores e os 5 caracteres posteriores

matches[0], matches[0][1], matches[0][2] #Posição inicial e posição final


from IPython.core.display import HTML #Gerar codigos em HTML

texto = ""
numero_palavras = 50 #Quantas palavras antes e quantas palavras depois
doc = pln(conteudo)
matches = matcher(doc)

display(HTML(f"<h1>{string.upper()} </h1>")) #<h1> = Titulo
display(HTML(f"""<p><strong>Resultados encontrados: </strong>{len(matches)}</p> """)) #<p> = paragrafo, <strong> = negrito
for i in matches:
    inicio = i[1] - numero_palavras
    if inicio < 0:
        inicio = 0    
    texto += str(doc[inicio:i[2] + numero_palavras]).replace(string, f"<mark>{string}</mark>")
    texto += "<br /><br />" #Pular linhas
display(HTML(f"""... {texto} ..."""))
