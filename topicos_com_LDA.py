Trabalho apresentado na disciplina de Processamento de Linguagem Natural
Pós Graduação PUC MINAS - Big Data e Dat Scienc

PROJETO de FINAL DE CURSO
"""

Resumo do trabalho:

Utilizar técnicas de análise de linguagem natural para verificar a similaridade (ou a diferença) 
de seis conceitos de SONOLOGIA capturados na internet.

Justificativa:

Conceituar é realizar o trabalho intelectual de apresentar as características essências de um ser, 
um objeto ou uma ideia. De acordo com os diversos ramos da ciência, pelo uso de metodologias 
diversificadas um mesmo objeto pode ser conceituado de várias formas. Entretanto, por maiores que sejam 
as diferenças, há um ponto comum ou um cerne que pode ser percebido de forma a permitir que as 
diversas ciências reconheçam estar tratando do mesmo objeto quando fazem referência ao conceito dele. 
Pretende-se com o uso do processamento de linguagem natural verificar o quão diversos conceitos de um mesmo 
objeto são similares ou diferentes, de acordo com o ramo da ciência que o adota ou pesquisadores. 
O presente trabalho analisa o conceito de SONOLOGIA, que é compartilhado pelas áreas de Engenharia, 
Música e Informática, podendo incluir também a área de Medicina.
"""

# Importações

import pandas as pd
import re
import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords
import string
from unidecode import unidecode
import gensim
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora

# Pré-preparação dos textos
# Pré-preparação do corpus em português

texto_1 = 'O termo é também usado para descrever a pesquisa interdisciplinar no campo da música eletrônica e da música computadorizada, utilizando disciplinas como acústica, eletrônica, informática, composição e psicoacústica. Este sentido do termo é amplamente associado ao Instituto de Sonologia, que foi estabelecido pelo compositor Gottfried Michael Koenig na Universidade de Utrecht em 1967 e mais tarde se mudou para o Conservatório Real de Haia em 1986. O termo também foi adotado para descrever o estudo da música eletrônica em outras instituições, incluindo o Centro de Sonologia Computacional (agora "Sound and Music Computing") na Universidade de Pádua, o Kunitachi College of Music, em Tóquio, a Faculdade de Música da Catalunha em Barcelona, e a Universidade Federal de Minas Gerais, no Brasil.' 
texto_2 = 'O termo vem sendo adotado por pesquisadores e instituições no Brasil para fazer referência a um campo híbrido de pesquisas musicais em que o som serve como elemento catalisador. Entre as áreas relacionadas estão as práticas eletroacústicas, as aplicações de novas tecnologias à produção e análise musical, a acústica e a psicoacústica, as musicologias interessadas em explorar aspectos estéticos e técnicos do som no contexto musical e os processos de criação multimidiáticos entre outras.'
texto_3 = 'Há cerca de 10 anos, diversos pesquisadores e instituições adotaram no Brasil o termo sonologia numa acepção muito próxima dos estudos do som. Este neologismo já foi empregado em diversos outros contextos, entre os quais o mais conhecido é o Instituto de Sonologia criado por Gottfried Michael Koenig na Universidade de Utrecht em 1967, e transferido posteriormente para o Conservatório Real de Haia em 1986. Embora sonologia e sound studies tenham o som como objeto central, a sonologia geralmente esteve voltada para aspectos mais técnicas da produção musical (síntese sonora, criação de interfaces e instrumentos eletrônicos, computação aplicada à música, acústica musical e psicoacústica). Por sua vez, os sound studies tendem a tomar o som num sentido que transcende a música, aproximando-se de disciplinas das ciências humanas como a sociologia, a antropologia e a filosofia. Como aponta Jonathan Sterne, os estudos do som promovem a análise de práticas sonoras e dos discursos e instituições que as descrevem. Com isso, redescrevem “o que o som produz no mundo das pessoas e o que as pessoas produzem no mundo sonoro” (Sterne, 2012: 2). No Brasil a sonologia adotou uma posição intermediaria, abarcando tanto o estudo crítico, analítico e reflexivo a respeitos das práticas sonoras, quanto se envolvendo com os aspectos criativos e técnicos dessas práticas.'
texto_4 ='Um dos pilares importantes para a institucionalização desse campo de estudo no país foi o reconhecimento e a criação da área de sonologia. Esta área não é computação musical, mas se beneficia das ferramentas da computação musical e, sobretudo, traz centralidade para o pensamento musical orientado para o som, que vai além das possibilidades da música baseada em notas musicais ou, ainda, uma música que é som. Essa sutileza de interpretação fez avançar tanto o processo criativo quanto a análise musical. “A sonologia abriu espaço para essa investigação e dois vetores epistêmicos de pesquisa se posicionaram: o processo criativo com suporte computacional e a análise musical com suporte computacional. São disciplinas que se colaboram, mas com certa autonomia”, elucida Manzolli.'
texto_5 = 'Também nos estudos de etnomusicologia ou de musicologias não ocidentais, a discussão sobre a experiência do som vem precedendo a discussão sobre música, sendo considerada o seu fundamento. Porém, o que se pode notar nas décadas de final do século XX e de início do XXI é que mesmo a recém-inaugurada área de sonologia em muitas pesquisas parece ainda cair nas mesmas armadilhas dualistas metafóricas, as quais se descreveram aqui para a música tonal. Muito do que se considera como objeto de estudo na área de sonologia refere-se a representações visuais daquilo que se escuta (partituras, cifras, sonogramas).'
texto_6 = 'Dessa nova abordagem sobre os estudos musicológicos surgiu o conceito de “sonologia”. Criado no Brasil em 2006, o termo trata da reflexão sobre processos interativos de produção musical levando-se em consideração os recursos tecnológicos da chamada computer music. “Essa é uma área de pesquisa bastante recente, que, além da forte conexão com o fazer musical por meio do computador, também está ligada aos programas de análise e de síntese sonora, hoje responsáveis pelas recomendações musicais que recebemos, por exemplo, em nossos smartphones”, diz Iazzetta. '

# Limpeza dos textos

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

texto_1_prep = pre_processamento_texto(texto_1)
texto_2_prep = pre_processamento_texto(texto_2)
texto_3_prep = pre_processamento_texto(texto_3)
texto_4_prep = pre_processamento_texto(texto_4)
texto_5_prep = pre_processamento_texto(texto_5)
texto_6_prep = pre_processamento_texto(texto_6)

# Comparação do tamanho dos corpus
# Tamanho de cada corpus

Tamanho_texto1 = len(texto_1_prep)
Tamanho_texto2 = len(texto_2_prep)
Tamanho_texto3 = len(texto_3_prep)
Tamanho_texto4 = len(texto_4_prep)
Tamanho_texto5 = len(texto_5_prep)
Tamanho_texto6 = len(texto_6_prep)
print(' Tamanho do texto 1: ', Tamanho_texto1, '\n',
     'Tamanho do texto 2: ', Tamanho_texto2, '\n', 
     'Tamanho do texto 3: ', Tamanho_texto3, '\n',
     'Tamanho do texto 4: ', Tamanho_texto4, '\n',
     'Tamanho do texto 5: ', Tamanho_texto5, '\n',
     'Tamanho do texto 6: ', Tamanho_texto6, '\n') 

# Tamanho proporcional ao total.

Tamanho_total = Tamanho_texto1 + Tamanho_texto2 + Tamanho_texto3 + Tamanho_texto4 + Tamanho_texto5 + Tamanho_texto6 
Por_texto1 = float((Tamanho_texto1/Tamanho_total) * 100)
Por_texto2 = ((Tamanho_texto2/Tamanho_total) * 100)
Por_texto3 = ((Tamanho_texto3/Tamanho_total) * 100)
Por_texto4 = ((Tamanho_texto4/Tamanho_total) * 100)
Por_texto5 = ((Tamanho_texto5/Tamanho_total) * 100)
Por_texto6 = ((Tamanho_texto6/Tamanho_total) * 100)
print(' Porcentagem do texto 1: ', "%.2f" % round(Por_texto1, 2), '%', '\n',
      'Porcentagem do texto 2: ', "%.2f" % round(Por_texto2, 2), '%','\n', 
      'Porcentagem do texto 3: ', "%.2f" % round(Por_texto3, 2), '%','\n', 
      'Porcentagem do texto 4: ', "%.2f" % round(Por_texto4, 2), '%','\n', 
      'Porcentagem do texto 5: ', "%.2f" % round(Por_texto5, 2), '%','\n', 
      'Porcentagem do texto 6: ', "%.2f" % round(Por_texto6, 2), '%','\n')
# BOW

vect_bag = CountVectorizer(binary=True) 
vetor_bow = vect_bag.fit_transform([' '.join(texto_1_prep),
                                   ' '.join(texto_2_prep),
                                   ' '.join(texto_3_prep),
                                   ' '.join(texto_4_prep), 
                                   ' '.join(texto_5_prep),
                                   ' '.join(texto_6_prep)])

texto1_bow = vetor_bow.todense()[0]
texto2_bow = vetor_bow.todense()[1]
texto3_bow = vetor_bow.todense()[2]
texto4_bow = vetor_bow.todense()[3]
texto5_bow = vetor_bow.todense()[4]
texto6_bow = vetor_bow.todense()[5]

# Embedding

word_vectors = gensim.models.KeyedVectors.load_word2vec_format('ptwiki_20180420_100d.txt', binary=False)
def get_word_vectors(frase):
    return np.mean(np.array([word_vectors[palavra] for palavra in frase if palavra in word_vectors.vocab]), axis = 0)

array_texto1 = get_word_vectors(texto_1_prep)
array_texto2 = get_word_vectors(texto_2_prep)
array_texto3 = get_word_vectors(texto_3_prep)
array_texto4 = get_word_vectors(texto_4_prep)
array_texto5 = get_word_vectors(texto_5_prep)
array_texto6 = get_word_vectors(texto_6_prep)

# Similaridade de Cosseno
# Com BOW

print('Similaridade do cosseno entre o conceito 1 e 2: ', cosine_similarity(texto1_bow, texto2_bow).round(2))
print('Similaridade do cosseno entre o conceito 1 e 3: ', cosine_similarity(texto1_bow, texto3_bow).round(2))
print('Similaridade do cosseno entre o conceito 1 e 4: ', cosine_similarity(texto1_bow, texto4_bow).round(2))
print('Similaridade do cosseno entre o conceito 1 e 5: ', cosine_similarity(texto1_bow, texto5_bow).round(2))
print('Similaridade do cosseno entre o conceito 1 e 6: ', cosine_similarity(texto1_bow, texto6_bow).round(2))

print('Similaridade do cosseno entre o conceito 2 e 3: ', cosine_similarity(texto2_bow, texto3_bow).round(2))
print('Similaridade do cosseno entre o conceito 2 e 4: ', cosine_similarity(texto2_bow, texto4_bow).round(2))
print('Similaridade do cosseno entre o conceito 2 e 5: ', cosine_similarity(texto2_bow, texto5_bow).round(2))
print('Similaridade do cosseno entre o conceito 2 e 6: ', cosine_similarity(texto2_bow, texto6_bow).round(2))

print('Similaridade do cosseno entre o conceito 3 e 4: ', cosine_similarity(texto3_bow, texto4_bow).round(2))
print('Similaridade do cosseno entre o conceito 3 e 5: ', cosine_similarity(texto3_bow, texto5_bow).round(2))
print('Similaridade do cosseno entre o conceito 3 e 6: ', cosine_similarity(texto3_bow, texto6_bow).round(2))

print('Similaridade do cosseno entre o conceito 4 e 5: ', cosine_similarity(texto4_bow, texto5_bow).round(2))
print('Similaridade do cosseno entre o conceito 4 e 6: ', cosine_similarity(texto4_bow, texto6_bow).round(2))

print('Similaridade do cosseno entre o conceito 5 e 6: ', cosine_similarity(texto5_bow, texto6_bow).round(2))

# Com Embedding

print('Similaridade do cosseno entre a fase 1 e 2: ', 1 - (distance.cosine(array_texto1, array_texto2).round(2)))
print('Similaridade do cosseno entre a fase 1 e 3: ', 1 - (distance.cosine(array_texto1, array_texto3).round(2)))
print('Similaridade do cosseno entre a fase 1 e 4: ', 1 - (distance.cosine(array_texto1, array_texto4).round(2)))
print('Similaridade do cosseno entre a fase 1 e 5: ', 1 - (distance.cosine(array_texto1, array_texto5).round(2)))
print('Similaridade do cosseno entre a fase 1 e 6: ', 1 - (distance.cosine(array_texto1, array_texto6).round(2)))

print('Similaridade do cosseno entre a fase 2 e 3: ', 1 - (distance.cosine(array_texto2, array_texto3).round(2)))
print('Similaridade do cosseno entre a fase 2 e 4: ', 1 - (distance.cosine(array_texto2, array_texto4).round(2)))
print('Similaridade do cosseno entre a fase 2 e 5: ', 1 - (distance.cosine(array_texto2, array_texto5).round(2)))
print('Similaridade do cosseno entre a fase 2 e 6: ', 1 - (distance.cosine(array_texto2, array_texto6).round(2)))

print('Similaridade do cosseno entre a fase 3 e 4: ', 1 - (distance.cosine(array_texto3, array_texto4).round(2)))
print('Similaridade do cosseno entre a fase 3 e 5: ', 1 - (distance.cosine(array_texto3, array_texto5).round(2)))
print('Similaridade do cosseno entre a fase 3 e 6: ', 1 - (distance.cosine(array_texto3, array_texto6).round(2)))

print('Similaridade do cosseno entre a fase 4 e 5: ', 1 - (distance.cosine(array_texto4, array_texto5).round(2)))
print('Similaridade do cosseno entre a fase 4 e 6: ', 1 - (distance.cosine(array_texto4, array_texto6).round(2)))

print('Similaridade do cosseno entre a fase 5 e 6: ', 1 - (distance.cosine(array_texto5, array_texto6).round(2)))

# Similaridade WMD

print('Distância entre os conceitos 1 e 2: ', word_vectors.wmdistance(texto_1_prep, texto_2_prep))
print('Distância entre os conceitos 1 e 3: ', word_vectors.wmdistance(texto_1_prep, texto_3_prep))
print('Distância entre os conceitos 1 e 4: ', word_vectors.wmdistance(texto_1_prep, texto_4_prep))
print('Distância entre os conceitos 1 e 5: ', word_vectors.wmdistance(texto_1_prep, texto_5_prep))
print('Distância entre os conceitos 1 e 6: ', word_vectors.wmdistance(texto_1_prep, texto_6_prep))

print('Distância entre os conceitos 2 e 3: ', word_vectors.wmdistance(texto_2_prep, texto_3_prep))
print('Distância entre os conceitos 2 e 4: ', word_vectors.wmdistance(texto_2_prep, texto_4_prep))
print('Distância entre os conceitos 2 e 5: ', word_vectors.wmdistance(texto_2_prep, texto_5_prep))
print('Distância entre os conceitos 2 e 6: ', word_vectors.wmdistance(texto_2_prep, texto_6_prep))

print('Distância entre os conceitos 3 e 4: ', word_vectors.wmdistance(texto_3_prep, texto_4_prep))
print('Distância entre os conceitos 3 e 5: ', word_vectors.wmdistance(texto_3_prep, texto_5_prep))
print('Distância entre os conceitos 3 e 6: ', word_vectors.wmdistance(texto_3_prep, texto_6_prep))

print('Distância entre os conceitos 4 e 5: ', word_vectors.wmdistance(texto_4_prep, texto_5_prep))
print('Distância entre os conceitos 4 e 6: ', word_vectors.wmdistance(texto_4_prep, texto_6_prep))

print('Distância entre os conceitos 5 e 6: ', word_vectors.wmdistance(texto_5_prep, texto_6_prep))

"""
Conclusões

Para o estudo de um conceito fica melhor o uso do contexto das palavras a fim de se verificar a similaridade. 
Os melhores resultados foram para o uso do embedding e do WMD. Os textos são semelhantes e 
isso significa que há um cerne no conceito de SONOLOGIA.
A pesquisa com similaridade por cosseno e BOW apontou que as palavras utilizadas são diferentes. 
Isso pode acontecer por ser um conceito para por pelo menos três áreas do saber, podendo ser quatro.
"""

# Análise de sentimentos dos conceitos com VADER

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Traduções

Texto_1 = 'The term is also used to describe interdisciplinary research in the field of electronic music and computerized music, using disciplines such as acoustics, electronics, computers, composition and psychoacoustics. This sense of the term is widely associated with the Institute of Sonology, which was established by the composer Gottfried Michael Koenig at Utrecht University in 1967 and later moved to the Royal Conservatory in The Hague in 1986. The term was also adopted to describe the study of electronic music at other institutions, including the Center for Computational Sonology (now "Sound and Music Computing") at the University of Padua, the Kunitachi College of Music in Tokyo, the Faculty of Music of Catalonia in Barcelona, and the Federal University of Minas Gerais, in Brazil.'
Texto_2 = 'The term has been adopted by researchers and institutions in Brazil to refer to a hybrid field of musical research in which sound serves as a catalyst. Among the related areas are electroacoustic practices, the application of new technologies to musical production and analysis, acoustics and psychoacoustics, musicologies interested in exploring aesthetic and technical aspects of sound in the musical context and the processes of multimedia creation, among others.'
Texto_3 = 'About 10 years ago, several researchers and institutions in Brazil adopted the term sonology in a sense very close to sound studies. This neologism has already been used in several other contexts, among which the best known is the Institute of Sonology created by Gottfried Michael Koenig at the University of Utrecht in 1967, and later transferred to the Royal Conservatory in The Hague in 1986. Although sonology and sound studies have sound as a central object, sonology has generally focused on more technical aspects of music production (sound synthesis, creation of interfaces and electronic instruments, computing applied to music, musical acoustics and psychoacoustics). In turn, sound studies tend to take sound in a sense that transcends music, approaching disciplines in the humanities such as sociology, anthropology and philosophy. As Jonathan Sterne points out, sound studies promote the analysis of sound practices and the discourses and institutions that describe them. With that, they rewrite “what sound produces in the world of people and what people produce in the world of sound” (Sterne, 2012: 2). In Brazil, sonology has adopted an intermediate position, encompassing both the critical, analytical and reflective study regarding sound practices, as well as being involved with the creative and technical aspects of these practices.'
Texto_4 = 'One of the important pillars for the institutionalization of this field of study in the country was the recognition and creation of the sonology area. This area is not musical computing, but it benefits from the tools of musical computing and, above all, brings centrality to musical thinking oriented to sound, which goes beyond the possibilities of music based on musical notes or, still, music that is sound . This subtlety of interpretation has advanced both the creative process and musical analysis. “Sonology opened space for this investigation and two epistemic research vectors were positioned: the creative process with computational support and the musical analysis with computational support. These are disciplines that collaborate, but with a certain autonomy ”, explains Manzolli.'
Texto_5 = 'Also in studies of ethnomusicology or non-Western musicologies, the discussion about the experience of sound has preceded the discussion about music, being considered its foundation. However, what can be seen in the decades of the end of the 20th century and the beginning of the 21st is that even the newly opened area of sonology in many researches still seems to fall into the same metaphorical dualistic traps, which have been described here for tonal music. . Much of what is considered as an object of study in the field of sonology refers to visual representations of what is heard (scores, figures, sonograms).'
Texto_6 = 'From this new approach to musicological studies, the concept of “sonology” emerged. Created in Brazil in 2006, the term deals with the reflection on interactive processes of musical production taking into account the technological resources of the so-called computer music. “This is a very recent area of research, which, in addition to the strong connection with making music through the computer, is also linked to the analysis and sound synthesis programs, which today are responsible for the musical recommendations we receive, for example, in our smartphones, ”says Iazzetta.'

# Análise com VADER

print('Análise de sentimentos para o texto 1: ', analyzer.polarity_scores(texto_1))
print('Análise de sentimentos para o texto 2: ', analyzer.polarity_scores(texto_2))
print('Análise de sentimentos para o texto 3: ', analyzer.polarity_scores(texto_3))
print('Análise de sentimentos para o texto 4: ', analyzer.polarity_scores(texto_4))
print('Análise de sentimentos para o texto 5: ', analyzer.polarity_scores(texto_5))
print('Análise de sentimentos para o texto 6: ', analyzer.polarity_scores(texto_6))

"""
Ficou claro que todos os textos são neutros. Isso era esperado porque conceitos são textos científicos. 
Não devem apresentar caráter positivo ou negativo, como apresentaria um texto de crítica, por exemplo.
"""

# Extração de tópicos com LDA

LDA = gensim.models.ldamodel.LdaModel

dicio = {}
dicio['texto_1'] = [texto_1]
dicio['texto_2'] = [texto_2]
dicio['texto_3'] = [texto_3]
dicio['texto_4'] = [texto_4]
dicio['texto_5'] = [texto_5]
dicio['texto_6'] = [texto_6]

df = pd.DataFrame(data = dicio)

essentials = df.values
essentials_clean = [pre_processamento_texto(str(e)) for e in essentials]
dictionary = corpora.Dictionary(essentials_clean)
doc_bow = [dictionary.doc2bow(doc) for doc in essentials_clean]
ldamodel = LDA(doc_bow, num_topics=2, id2word = dictionary, passes=1000, random_state=123, alpha='auto',per_word_topics=True)
ldamodel.show_topics()

"""
Foram pesquisados a princípio quatro tópicos. O resultado ficou com três 
tópicos iguais e um diferente. Então foi feita a opação por dois tópicos apenas.

Definições para os tópicos encontrados:

Tópico 0: Pesquisa em música ou partitura.
Tópico 1: A sonologia no Brasil estuda som e música

Na análise de tópicos ficou claro que todos os conceitos se referem a SONOLOGIA 
como uma área, que no Brasil, pesquisa som e música. Este provavelmente é o cerne do conceito.
"""