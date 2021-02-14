"""
Trabalho apresentado na disciplina de Processamento de Linguagem Natural
Pós Graduação PUC MINAS - Big Data e Dat Scienc

SIMILARIDADE
"""

!pip install afinn
!python -m textblob.download_corpora
!pip install -U textblob
!pip install vaderSentiment
!pip install pyemd

import re
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
import string
from unidecode import unidecode
import pandas as pd
import bz2
import gensim
import warnings
import numpy as np
from gensim.models import word2vec
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from tqdm._tqdm_notebook import tqdm_notebook
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

warnings.filterwarnings('ignore')
tqdm_notebook.pandas()

import matplotlib.pyplot as plt
%matplotlib inline

# Carregando os embeddings
word_vectors = gensim.models.KeyedVectors.load_word2vec_format('ptwiki_20180420_100d.txt', binary=False)

"""
SIMILARIDADE DOS DOCUMENTOS
"""

frase1 = "Excelente produto chegou antes do prazo indico e recomendo produto bom pois já testei e foi mais que aprovado" 
frase2 = "SUPER RECOMENDO, PREÇO, QUALIDADE #BRASTEMP, EFICIÊNCIA NA ENTREGA, E FACILIDADE DE PAGAMENTO. MUITO BOM!!!"
frase3 = "A tampa do fogão veio com problemas com o pino de encaixe solto e precisa de reparos"
frase4 = "Fogão ótimo!"

def pre_processamento_texto(corpus):
    
    
    #Pre processa o texto retornando o corpus processado
    #Tokenização
    documento_limpo = re.findall(r'\w+(?:\w=)?|[^\w\s]', corpus)
   
    #Remoção de capitalização
    documento_limpo = [palavra.lower() for palavra in documento_limpo]
        
    #Remoção de stopwords
    lista = stopwords.words('portuguese')
    documento_limpo = [palavra for palavra in documento_limpo if palavra not in lista] 

    #Remoção de números
    documento_limpo = [re.sub('\d+', '', palavra) for palavra in documento_limpo]

    #Remoção de acentuação
    documento_limpo = [re.sub('\d+', '', palavra) for palavra in documento_limpo]

    #Remoção de pontuação
    lista = string.punctuation
    documento_limpo = [palavra for palavra in documento_limpo if palavra not in lista]

    #Remoção de acentuação
    documento_limpo = [unidecode(palavra) for palavra in documento_limpo]
    
    return documento_limpo

frase1_limpa = pre_processamento_texto(frase1)
frase2_limpa = pre_processamento_texto(frase2)
frase3_limpa = pre_processamento_texto(frase3)
frase4_limpa = pre_processamento_texto(frase4)

# Distância de Jaccard
def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return (float(intersection) / union)

print(jaccard(frase1_limpa, frase2_limpa))
print(jaccard(frase1_limpa, frase3_limpa))
print(jaccard(frase2_limpa, frase3_limpa))
print(jaccard(frase1_limpa, frase4_limpa))

# Distância de Cosseno
# BOW - Distância do cosseno

vetor_bow = vect_bag.fit_transform([' '.join(frase1_limpa),
                                   ' '.join(frase2_limpa),
                                   ' '.join(frase3_limpa),
                                   ' '.join(frase4_limpa)])
frase1_bow = vetor_bow.todense()[0]
frase2_bow = vetor_bow.todense()[1]
frase3_bow = vetor_bow.todense()[2]
frase4_bow = vetor_bow.todense()[3]

print('Distância do cosseno entre a fase 1 e 2: ', distance.cosine(frase1_bow, frase2_bow))
print('Distância do cosseno entre a fase 1 e 3: ', distance.cosine(frase1_bow, frase3_bow))
print('Distância do cosseno entre a fase 2 e 3: ', distance.cosine(frase2_bow, frase3_bow))
print('Distância do cosseno entre a fase 1 e 4: ', distance.cosine(frase1_bow, frase4_bow))

from sklearn.metrics import pairwise

print('Similaridade do cosseno entre a fase 1 e 2: ', cosine_similarity(frase1_bow, frase2_bow))
print('Similaridade do cosseno entre a fase 1 e 3: ', cosine_similarity(frase1_bow, frase3_bow))
print('Similaridade do cosseno entre a fase 2 e 3: ', cosine_similarity(frase2_bow, frase3_bow))
print('Similaridade do cosseno entre a fase 1 e 4: ', cosine_similarity(frase1_bow, frase4_bow))

# Embedding - Distância do cosseno

def get_word_vectors(frase):
    return np.mean(np.array([word_vectors[palavra] for palavra in frase if palavra in word_vectors.vocab]), axis = 0)

array_frase1 = get_word_vectors(frase1_limpa)
array_frase2 = get_word_vectors(frase2_limpa)
array_frase3 = get_word_vectors(frase3_limpa)
array_frase4 = get_word_vectors(frase4_limpa)

print('Distância do cosseno entre a fase 1 e 2: ', distance.cosine(array_frase1, array_frase2))
print('Distância do cosseno entre a fase 1 e 3: ', distance.cosine(array_frase1, array_frase3))
print('Distância do cosseno entre a fase 2 e 3: ', distance.cosine(array_frase2, array_frase3))
print('Distância do cosseno entre a fase 1 e 4: ', distance.cosine(array_frase1, array_frase4))

# WMD

print('Distância entre as frases 1 e 2: ', word_vectors.wmdistance(frase1_limpa, frase2_limpa))
print('Distância entre as frases 1 e 3: ', word_vectors.wmdistance(frase1_limpa, frase3_limpa))
print('Distância entre as frases 2 e 3: ', word_vectors.wmdistance(frase2_limpa, frase3_limpa))
print('Distância entre as frases 1 e 4: ', word_vectors.wmdistance(frase1_limpa, frase4_limpa))

# Classificação de Documentos
# Dataset

df = pd.read_csv("imdb-reviews-pt-br.csv")
target = df['sentiment'].replace(['neg', 'pos'], [0 ,1])

def pre_processamento_texto_return_str(corpus):
    
    
    #Pre processa o texto retornando o corpus processado
    #Tokenização
    documento_limpo = re.findall(r'\w+(?:\w=)?|[^\w\s]', corpus)
   
    #Remoção de capitalização
    documento_limpo = [palavra.lower() for palavra in documento_limpo]
        
    #Remoção de stopwords
    lista = stopwords.words('portuguese')
    documento_limpo = [palavra for palavra in documento_limpo if palavra not in lista] 

    #Remoção de números
    documento_limpo = [re.sub('\d+', '', palavra) for palavra in documento_limpo]

    #Remoção de acentuação
    documento_limpo = [re.sub('\d+', '', palavra) for palavra in documento_limpo]

    #Remoção de pontuação
    lista = string.punctuation
    documento_limpo = [palavra for palavra in documento_limpo if palavra not in lista]

    #Remoção de acentuação
    documento_limpo = [unidecode(palavra) for palavra in documento_limpo]
    
    documento_limpo = ' '.join( documento_limpo)
    
    return documento_limpo.lower()
df['text_pt_limpo'] = df['text_pt'].progress_apply(lambda x: 
                                                   pre_processamento_texto_return_str(x))
vect = CountVectorizer()
vetor_bow = vect.fit_transform(df['text_pt_limpo'])

# Embedding

def pre_processamento_texto_token(corpus):
    
    
    #Pre processa o texto retornando o corpus processado
    #Tokenização
    documento_limpo = re.findall(r'\w+(?:\w=)?|[^\w\s]', corpus)
   
    #Remoção de capitalização
    documento_limpo = [palavra.lower() for palavra in documento_limpo]
        
    #Remoção de stopwords
    lista = stopwords.words('portuguese')
    documento_limpo = [palavra for palavra in documento_limpo if palavra not in lista] 

    #Remoção de números
    documento_limpo = [re.sub('\d+', '', palavra) for palavra in documento_limpo]

    #Remoção de acentuação
    documento_limpo = [re.sub('\d+', '', palavra) for palavra in documento_limpo]

    #Remoção de pontuação
    lista = string.punctuation
    documento_limpo = [palavra for palavra in documento_limpo if palavra not in lista]

    #Remoção de acentuação
    documento_limpo = [unidecode(palavra) for palavra in documento_limpo]
    
    return documento_limpo

df['text_pt_token'] = df['text_pt'].progress_apply(lambda x: 
                                                   pre_processamento_texto_token(x))
x_embedding = df['text_pt_token'].progress_apply(lambda x: get_word_vectors(x))

# CountVectorizer
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(vetor_bow, target,random_state=123)
modelo_bow = LogisticRegression()
modelo_bow.fit(X_train_bow, y_train_bow)
y_predito = modelo_bow.predict(X_test_bow)
print(classification_report(y_test_bow, y_predito))

# Embedding - Outra forma
X_train_emb, X_test_emb, y_train_emb, y_test_emb = train_test_split(x_embedding.values, target,random_state=123)
modelo_embedding = LogisticRegression()
X_train_emb = pd.DataFrame([x for x in X_train_emb])
modelo_embedding.fit(X_train_emb, y_train_emb)
X_test_emb = pd.DataFrame([x for x in X_test_emb])
y_predito = modelo_embedding.predict(X_test_emb)
print(classification_report(y_test_emb, y_predito))

# Análise de sentimentos

texto_neg = df.loc[0, "text_en"]
texto_pos = df.loc[49431, "text_en"]

# Vader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
analyzer.polarity_scores(texto_neg)
analyzer.polarity_scores(texto_pos)

# TextBlob
from textblob import TextBlob
sentence=TextBlob(texto_neg)
sentence.sentiment
sentence=TextBlob(texto_pos)
sentence.sentiment

# Afinn
from afinn import Afinn
afinn = Afinn()
afinn.score(texto_pos)
afinn = Afinn()
afinn.score(texto_neg)

# Spacy - Textos em inglês
!python -m spacy download en_core_web_md

import spacy
import pandas as pd

nlp = spacy.load('en_core_web_md')
doc = nlp("This is some text that I am processing with Spacy")

def calcula_media_posicao(x):
    soma = 0
    vector = []
    for i in range(0,len(doc)):
        vector.append(doc[i].vector)    
    
    for v in vector:
        soma += v[x]
    return soma/len(doc)

round(calcula_media_posicao(10),6)
round(doc.vector[10], 6)