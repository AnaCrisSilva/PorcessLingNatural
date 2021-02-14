"""
Trabalho apresentado na disciplina de Processamento de Linguagem Natural
Pós Graduação PUC MINAS - Big Data e Dat Scienc

POS TAGGING
"""

# Abordagem Manual

def print_found_groups(pattern, phrase):
    regex = re.compile(pattern)
    for m in regex.finditer(phrase):
        print({k:v for(k,v) in m.groupdict().items() if v is not None})

phrase1 = "eu estudo, você estuda, nós estudamos e vc sabe como é... estudos em todo lugar "
estudar_pattern_try1 = r'(?P<VERB_ESTUDAR>\bestud\w?\b)'
print_found_groups(estudar_pattern_try1, phrase1)

def get_found_groups(pattern, phrase):
    regex = re.compile(pattern)
    return {v:k for m in regex.finditer(phrase) for(k,v) in m.groupdict().items() if v is not None }

frase = "O principal uso que fazemos do canvas é ver e entregar exercícios, além de acessar os materiais postados pelos professores. É importante realizar o acesso todos os dias."
Lista_verbos = ['uso', 'faz\w{1,4}', 'é', 'ver', 'entregar', 'acess\w{1,2}', 'postados', 'realizar']
exper_verbos = r'(?P<VERB>' + '|'.join(Lista_verbos) + ')'
get_found_groups(exper_verbos,frase)

def get_found_groups(pattern, phrase):
    regex = re.compile(pattern)
    return {v:k for m in regex.finditer(phrase) for(k,v) in m.groupdict().items() if v is not None }

lista_substantivo = ['principal', 'canvas', 'exercício\W{0,1}', 'materia\W{0,2}', 'professores', 'acesso', 'dias']
exper_subs = r'(?P<SUBSTANTIVO>' + '|'.join(lista_substantivo) + ')'
get_found_groups(exper_subs,frase)

# Mac_Morpho com NLTK
nltk.download('mac_morpho')
words_tagged = nltk.corpus.mac_morpho.tagged_words()
%%time
tags = {}

for palavra, tag in words_tagged:
    if tag not in tags.keys():
        tags[tag] = 1
    else: 
        tags[tag] += 1

tags_ordenadas = sorted(tags.items(), key = operator.itemgetter(1))

# POS Tagging com Spacy - Desambiguando frases

npl = pt_core_news_sm.load()

txt0="Ele foi andar"
txt1="O andar estava triste"

def analisa_txt(txt):
    npl_txt = npl(txt)
    i = 0
    for palavra in npl_txt:
        print('TOKEN: ', palavra.text)
        print('\tPostag: ', npl_txt[i].pos_)
        print('\tDependência: ', npl_txt[i].dep_)
        i += 1
    displacy.render(npl_txt, style='dep', jupyter=True)

analisa_txt(txt0)
analisa_txt(txt1)

txt2="Essa cola cola papel"

analisa_txt(txt2)

txt3="Segundo o que me disseram, ele estava triste"
txt4="No segundo dia de trabalho ela gastou menos"
analisa_txt(txt3)
analisa_txt(txt4)

txt5="Hora de fazer imposto de renda"
txt6="Aquilo foi imposto pelo chefe"
analisa_txt(txt5)
analisa_txt(txt5)

txt7="Todo dia eu caminho"
txt8="Esse é o caminho mais fácil"
analisa_txt(txt7)
analisa_txt(txt8)

# Named Entity Recognition

def match_class(target):                                                        
    def do_match(tag):                                                          
        classes = tag.get('class', [])                                          
        return all(c in classes for c in target)                                
    return do_match 

def get_text_url(url):
    res = requests.get(url)
    html = res.text
    soup = BeautifulSoup(html, 'html5lib')
    #remove marcações de scripts e style
    texto = soup.find_all(match_class(["content-text__container"]))
    all_text = ""
    for t in texto:
        all_text += t.get_text()
    return all_text

noticia = {}
texto_noticia = {i:get_text_url(noticia[i]) for i in noticia.keys()}

npl = pt_core_news_sm.load()
nlp_texto = {i: npl(texto_noticia[i]) for i in noticia.keys()}
ents = {i: set(map(str, nlp_texto[i].ents)) for i in noticia.keys()}

# Retreino NER

nlp = spacy.load("pt_core_news_sm")
texto = "João nasceu em Paris em 01/01/2000"
texto_nlp = nlp(texto)
displacy.render(texto_nlp, style='ent',jupyter=True)

train_data = [
("Em 04/12/1992 nasceu Joana", {'entities':[(3, 13, "DATE"), (21,26, "PER")]}),
("Data de início: 10/01/2018", {'entities':[(16, 26, "DATE")]}),
("Maria se mudou para Paris", {'entities':[(20,26, "LOC")]}),
("Paris cidade das luzes", {'entities':[(0,5, "LOC")]}),    
("Maria nasceu em Contagem no dia 07/05/2018", {'entities':[(0,5,"PER"),(16,24,"LOC"),(32,42, "DATE")]})
]

def train_spacy(data, iterations):
    TRAIN_DATA = data
    
    nlp = spacy.blank('pt')  # create blank Language class
    nlp.vocab.vectors.name = 'spacy_pretrained_vectors'
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
       

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)           
    return nlp

modelo_novo = train_spacy(train_data, 30)
type(modelo_novo)
print(spacy.__version__)
modelo_novo.to_disk("./modelo_novo_NER")