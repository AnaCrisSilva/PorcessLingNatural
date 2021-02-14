# -*- coding: utf-8 -*-
"""
Trabalho apresentado na disciplina de Processamento de Linguagem Natural
Pós Graduação PUC MINAS - Big Data e Dat Scienc

EXPRESSÕES REGULARES
"""

!pip install unidecode
!pip install wikipedia
!pip install spacy
!python -m spacy download en
!python -m spacy download pt

import re
import datetime

texto = "Vamos encontrar Padrões nesta string!! \nAgora é a nossa primeira prática de NLP!! Vamos aprender a procurar padrões!! \nBelo Horizonte, "+ str(datetime.datetime.now().date())+"."

# Uso do Match
re.match("Vamos", texto)
re.match("nesta", texto)
re.match(r"vamos", texto)

# Uso do Search
re.search("Vamos", texto)
re.search("encontrar", texto)
re.search("padrões", texto)
re.search("Padrões", texto)
re.search("padrões", texto, re.IGNORECASE) 

# Uso do Findall

re.findall("Vamos", texto)
re.findall("padrões", texto)
re.findall("padrões", texto, re.I)
re.findall(r'\w+a\w+', texto)
re.findall(r'\w*a\w*', texto)
re.findall(r'\w*a\w*', texto)

# Uso do Finditer
re.finditer("Vamos", texto)

res = re.finditer("Vamos", texto)
[r for r in res]

re.finditer("padrões", texto)
re.finditer("padrões", texto, re.I)

# Uso do Split
re.split('\n',texto)
texto.split("\s+")
(re.split(r'\s+',texto))

# Uso do Sub
re.sub('\w+a\w+', 'a-word', texto)

#Uso do Subn
re.subn('\w+a\w+', 'a-word', texto)

"""
Técnicas de Pré-processamento
"""

import nltk
import wikipedia
import re

from nltk.probability import FreqDist
nltk.download()

# Definindo o corpus

wikipedia.set_lang("pt")
pln = wikipedia.page("PLN")
corpus = pln.content
print(pln.content)

# Tokenização e Regex

tokens_split = corpus.split()
re.findall(r"\w+(?:'\w+)?|[^\w\s]", corpus)
tokens_regex = re.findall(r"\w+(?:'\w+)?|[^\w\s]", corpus)

# NLTK

tokens_nltk = nltk.word_tokenize(corpus)
plot_frequencia_tokens(tokens_split)
plot_frequencia_tokens(tokens_regex)
plot_frequencia_tokens(tokens_nltk)

# Capitalização
tokens = corpus.lower()

# Remoção Stopwords
from nltk.corpus import stopwords
portugues_stops = stopwords.words('portuguese')
tokens_sem_stop = portugues_stops

# Remoção Números
tokens_sem_numbers = re.sub('[0-9]','', corpus)

# Remoção Pontuação
import string
string.punctuation
tokens_sem_punction = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]','', corpus)
sorted(tokens_sem_punction)                            
                               

# Remoção Acentos
from unidecode import unidecode
tokens_sem_acentos = unidecode(tokens_sem_punction)
plot_frequencia_tokens(tokens_sem_acentos)

# Stemming
stemmer = nltk.stem.RSLPStemmer()
tokens_stemmer = stemmer.stem(tokens_sem_acentos)
plot_frequencia_tokens(tokens_stemmer)

Lemmatization
import spacy 
nlp = spacy.load('pt')
str_tokens = ''.join(tokens_sem_punction)
doc = nlp(str_tokens)
type(doc)
token_lemm = token_lemm.append(token) for token in doc
len(token_lemm)