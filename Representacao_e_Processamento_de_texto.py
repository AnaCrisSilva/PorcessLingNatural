"""
Trabalho apresentado na disciplina de Processamento de Linguagem Natural
Pós Graduação PUC MINAS - Big Data e Dat Scienc

REPRESENTAÇÃO E PROCESSAMENTO DE TEXTO
"""

!pip install gensim
!pip install umap-learn
!pip install wikipedia
!pip install unidecode

import re
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
import wikipedia
import string
from unidecode import unidecode
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import urllib.request
import bz2
import gensim
import warnings
import numpy as np
from gensim.models import word2vec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
#import umap
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
%matplotlib inline

# DEFINIÇÃO DO corpus
# Base

wikipedia.set_lang("pt")
bh = wikipedia.page("Belo Horizonte")
corpus = bh.content

documentos = \
["Belo Horizonte é um município brasileiro e a capital do estado de Minas Gerais",
"A populacao de Belo Horizonte é estimada em 2 501 576 habitantes, conforme estimativas do Instituto Brasileiro de Geografia e Estatística",
"Belo Horizonte já foi indicada pelo Population Crisis Commitee, da ONU, como a metrópole com melhor qualidade de vida na América Latina",
"Belo Horizonte é mundialmente conhecida e exerce significativa influência nacional e até internacional, seja do ponto de vista cultural, econômico ou político",
"Belo Horizonte é a capital do segundo estado mais populoso do Brasil, Minas Gerais"]

# Processamento

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
    lista2 = string.punctuation
    documento_limpo = [palavra for palavra in documento_limpo if palavra not in lista]

    #Remoção de acentuação
    documento_limpo = [unidecode(palavra) for palavra in documento_limpo]
    
    return documento_limpo 

corpus_processado = [pre_processamento_texto(doc) for doc in documentos]

# Representação Textual

# N-grams
# NLTK
corpus_ngrams_nltk = [list(ngrams(doc, 2)) for doc in corpus_processado]
for i in range(0, len(corpus_ngrams_nltk)):
    print('Bigrama', 1, ':')
    print(corpus_ngrams_nltk[i])
    print('____________________________________')

# Phrases - Gensim
model_corpus_phrases = gensim.models.Phrases(corpus_processado, min_count=1)
bigram_corpus = model_corpus_phrases[corpus_processado]
for i in range(0, len(bigram_corpus)):
    print('Bigrama', 1, ':')
    print(bigram_corpus[i])
    print('____________________________________')

# TD-IDF
vect = TfidfVectorizer()
docs_tdidf = vect.fit_transform(documentos)
pd.DataFrame(docs_tdidf.todense())
docs_tdidf.shape
print('Lista de features detectadas: \n', vect.get_feature_names())
print('Total de features detectadas: ', len(vect.get_feature_names()))
pd.DataFrame(docs_tdidf.todense().T, index = vect.get_feature_names())

vect = TfidfVectorizer()
vect.set_params(ngram_range=(1, 2))
vect.set_params(max_df=0.5)
vect.set_params(min_df=2)
docs_tdidf = vect.fit_transform(documentos)
pd.DataFrame(docs_tdidf.todense().T, index = vect.get_feature_names())

# Bag of Words
vect_bag = CountVectorizer(binary=True) 
vetor_bow = vect_bag.fit_transform(documentos)

pd.DataFrame(vetor_bow.todense().T, index = vect_bag.get_feature_names())
data = pd.DataFrame(vetor_bow.todense().T, index = vect_bag.get_feature_names())
data['soma'] = data.sum(axis = 1)
conta = data['soma'].value_counts().to_dict()

# Embedding
"""
newfilepath = "embedding_wiki_100d_pt.txt"
filepath = "ptwiki_20180420_100d.txt.bz2"
with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filepath, 'rb') as file:
    for data in iter(lambda : file.read(100 * 1024), b''):
        new_file.write(data)
"""

from nltk.corpus import machado
machado.fileids()
raw_casmurro = machado.raw('contos/macn001.txt')
machado_sents = machado.sents()
texto_limpo = [[palavra.lower() for palavra in sentence if palavra not in string.punctuation] for sentence in machado_sents]
model = word2vec.Word2Vec(texto_limpo, min_count=10, workers=4, seed=123, sg=1, size=300, window=5)
model['dom']
model.wv.save_word2vec_format('model_emb_treinado.bin', binary=True)

total_palavras = []

for p in palavras_origem:
    total_palavras.extend(palavras_similares(p))

print(total_palavras)

array_embeddings = np.empty((0,300), dtype = 'f')
for palavra in total_palavras:
    array_embeddings = np.append(array_embeddings, np.array([model[palavra]]), axis = 0)
def plot_embedding_2d(array_2d, all_words, words_seed):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    for (x, y), w in zip(pca_result, all_words):
        ax.scatter(x, y, c='red' if w in words_seed else 'blue')
        ax.annotate(w,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(array_embeddings)
plot_embedding_2d(pca_result, total_palavras, palavras_origem)

# TSNE
tsne = TSNE(n_components=2, random_state=0)
plot_embedding_2d(tsne_result, total_palavras, palavras_origem)

# UMAP
map = umap.UMAP()
plot_embedding_2d(umap_result, total_palavras, palavras_origem)