#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:56:18 2020

@author: simransetia
This program is used to construct merged network of video transcripts and wiki articles.
"""
import nltk
import matplotlib.pyplot as plt
f=open('/Users/simransetia/Documents/Dataset/JOCWIKI1/doc1/week1.txt')
corpus=f.readlines()


def preprocessing(corpus):
    # initialize
    clean_text = []

    for row in corpus:
        # tokenize
        tokens = nltk.tokenize.word_tokenize(row)
        # lowercase
        tokens = [token.lower() for token in tokens]
        # isword
        tokens = [token for token in tokens if token.isalpha()]
        clean_sentence = ''
        clean_sentence = ' '.join(token for token in tokens)
        clean_text.append(clean_sentence)
        
    return clean_text
    
all_text = preprocessing(corpus)
for each in all_text:
    if each=='':
        all_text.remove(each)
# sklearn countvectorizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# Convert a collection of text documents to a matrix of token counts
cv = CountVectorizer(ngram_range=(1,1), stop_words = 'english')
# matrix of token counts
X = cv.fit_transform(all_text)
Xc = (X.T * X) # matrix manipulation
Xc.setdiag(0)
Xc=np.array(Xc.toarray())
names = cv.get_feature_names()
namesd={}
i=0
for each in names:
    namesd[i]=each
    i=i+1
import networkx as nx
G=nx.from_numpy_matrix(Xc)
pos = nx.spring_layout(G)
nx.draw_networkx_labels(G,pos,namesd,font_size=6)


import nltk
f=open('/Users/simransetia/Documents/Dataset/video1.txt')
corpus=f.readlines()


def preprocessing(corpus):
    # initialize
    clean_text = []

    for row in corpus:
        # tokenize
        tokens = nltk.tokenize.word_tokenize(row)
        # lowercase
        tokens = [token.lower() for token in tokens]
        # isword
        tokens = [token for token in tokens if token.isalpha()]
        clean_sentence = ''
        clean_sentence = ' '.join(token for token in tokens)
        clean_text.append(clean_sentence)
        
    return clean_text
    
all_text = preprocessing(corpus)
for each in all_text:
    if each=='':
        all_text.remove(each)
# sklearn countvectorizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
# Convert a collection of text documents to a matrix of token counts
cv = CountVectorizer(ngram_range=(1,1), stop_words = 'english')
# matrix of token counts
X = cv.fit_transform(all_text)
Xc = (X.T * X) # matrix manipulation
Xc.setdiag(0)
Xc=np.array(Xc.toarray())
names1 = cv.get_feature_names()
namesd={}
i=0
for each in names1:
    namesd[i]=each
    i=i+1
import networkx as nx
G1=nx.from_numpy_matrix(Xc)
pos2 = nx.spring_layout(G1)
for k,v in pos2.items():
    # Shift the x values of every node by 10 to the right
    v[0] = v[0] +50
nx.draw_networkx_labels(G1,pos2,namesd,font_size=6)

#nx.draw(G)
#nx.draw(G1)
#plt.show()

#print(graph_tool.topology.mark_subgraph(G1, G))
lda_videos=['right',
 'programming',
 'really',
 'course',
 'program',
 'language',
 'people',
 'thing',
 'question',
 'learn',
 'try',
 'computer',
 'start',
 'water',
 'piece',
 'instructions',
 'cookie',
 'understand',
 'think',
 'written',
 'getting',
 'application',
 'using',
 'difficult',
 'coffee']
lda_article=['programming',
 'instructions',
 'computer',
 'program',
 'start',
 'discourage',
 'learn',
 'understand',
 'people',
 'written',
 'really',
 'language',
 'process',
 'phone',
 'piece',
 'prerequisites',
 'problem',
 'specific']
'''
lda_article=['statement',
 'download',
 'python',
 'spyder',
 'click',
 'variable',
 'indentation',
 'output',
 'console',
 'pane',
 'write',
 'execute',
 'using',
 'quote',
 'input',
 'please',
 'installation',
 'block',
 'programming',
 'instead',
 'unchecked',
 'program',
 'value',
 'watch',
 'language',
 'example',
 'single',
 'double',
 'terminal',
 'interpreter',
 'properly',
 'prototyping',
 'otherwise',
 'print',
 'press',
 'present',
 'platform',
 'permanently',
 'patient',
 'package',
 'option']
lda_videos=['print',
 'equal',
 'answer',
 'right',
 'three',
 'number',
 'enter',
 'python',
 'time',
 'write',
 'hello',
 'become',
 'display',
 'something',
 'execute',
 'thing',
 'command',
 'please',
 'happen',
 'going',
 'anaconda',
 'using',
 'programming',
 'correct',
 'input',
 'download',
 'simply',
 'understand',
 'mean',
 'whatever',
 'variable',
 'computer',
 'sudarshan',
 'start']
'''
common=[]
common1=[]
labels={}
labels1={}
newlabels1={}
newlabels={}
i=0
'''
for each in lda_article:
    for each1 in lda_videos:
        if each==each1:
            common.append(names.index(each))
            common1.append(names1.index(each1))
            labels[names.index(each)]=names
            labels1[names1.index(each1)]=names.index(each)
            i=i+1
'''
for each in lda_article:
    labels[names.index(each)]=each
    common.append(names.index(each))
for each in lda_videos:
    labels1[names1.index(each)]=each
    common1.append(names1.index(each))

for u,v in labels1.items():
    if v in labels.values():
        newlabels1[names1.index(v)]=v+" same"
    else:
        newlabels1[names1.index(v)]=v
for u,v in labels.items():
    if v in labels1.values():
        newlabels[names.index(v)]=v+" same"
    else:
        newlabels[names.index(v)]=v

H = G.subgraph(common)
H=nx.relabel_nodes(H,newlabels)
#H=nx.relabel_nodes(H,labels)
H1 = G1.subgraph(common1)  
print(H1.nodes())
H1=nx.relabel_nodes(H1,newlabels1)
print(H1.nodes())
