import bs4 as bs #Leitura e processamento de dados na web
import urllib.request
import nltk
import spacy


# Marcação POS = (Part=of=speech) atribuir para as palavras da fala, como substantivos, adjetivos e verbos
# Importante para a detecção de entidades no texto, pois primeiro é necessário saber o que o texto contém

pln = spacy.load("pt_core_news_sm")
documento = pln("Estou aprendendo processamento de linguagem natural, curos em Curitiba")

type(documento)

# https://spacy.io/api/data-formats#config
# https://www.sketchengine.eu/portuguese-freeling-part-of-speech-tagset/
for token in documento: #Percorrer cada uma das palavras
    print(token.text, token.pos_)
# Estou AUX
# aprendendo VERB
# processamento NOUN
# de ADP
# linguagem NOUN
# natural ADJ
# , PUNCT
# curos NOUN
# em ADP
# Curitiba PROPN


#Lematização e stemização
for token in documento:
    print(token.text, token.lemma_)
# Estou estar
# aprendendo aprender
# processamento processamento
# de de
# linguagem linguagem
# natural natural
# , ,
# curos curo
# em em
# Curitiba Curitiba

#Lematização extrair o radical das palavras
doc = pln("encontrei encontraram encotrarão encontrariam cursando curso cursei")
[token.lemma_ for token in doc]


nltk.download("rslp")

stemmer = nltk.stem.RSLPStemmer()
stemmer.stem("aprender")

for token in documento:
    print(token.text, token.lemma_, stemmer.stem(token.text))

#USAR MAIS A LEMATIZAÇÃO DO QUE A STEMIZAÇÃO
