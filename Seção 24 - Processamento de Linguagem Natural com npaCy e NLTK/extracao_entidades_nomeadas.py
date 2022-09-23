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


#----------- Extração de entidades nomeadas
# NER (Named-Entity Recognition)
# Encontrar e classificar entidades no texto, dependendo da base de dados utilizada para o treinamento (Pessoa, Localização, Empresa, Numéricos)
# Usado em chatbots para saber o assunto falado

for entidade in doc.ents:
    print(entidade.text, entidade.label_) #Encontrando as entidades.

from spacy import displacy
displacy.render(doc, style = "ent", jupyter=True)