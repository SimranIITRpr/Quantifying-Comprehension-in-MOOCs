#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:42:22 2020

@author: simransetia
This program is used to construct merged network of video transcripts and QnA forum.
"""
import nltk
#f=open('/Users/simransetia/Documents/Dataset/video1.txt')
f=open('/Users/simransetia/Documents/Dataset/JOCWIKI1/discuss/discussion1.txt')
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
#nx.draw(G1)
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
lda_videos=['letter',
 'movie',
 'player',
 'question',
 'character',
 'present',
 'value',
 'create',
 'write',
 'answer',
 'position',
 'number',
 'equal',
 'minus',
 'unlock',
 'thing',
 'condition',
 'particular',
 'square',
 'guess',
 'magic',
 'check',
 'space',
 'symbol',
 'display',
 'people',
 'define',
 'start',
 'person',
 'birth',
 'column',
 'generate',
 'comma',
 'month',
 'given',
 'point',
 'element',
 'want',
 'randomly',
 'append']
'''
lda_videos=['right',
 'programming',
 'something',
 'really',
 'course',
 'program',
 'language',
 'people',
 'whole',
 'thing',
 'question',
 'learn',
 'try',
 'computer',
 'start',
 'water',
 'piece',
 'instructions',
 'three',
 'cookie',
 'understand',
 'think',
 'written',
 'getting',
 'application',
 'using',
 'difficult',
 'coffee',
 'call']
lda_discuss=['month',
 'reply',
 'question',
 'ask',
 'assignment',
 'programming',
 'please',
 'course',
 'video',
 'thank',
 'scratch',
 'python',
 'command',
 'learn',
 'press',
 'edit',
 'problem',
 'language',
 'answer',
 'change',
 'download',
 'basic',
 'using',
 'logic',
 'output',
 'start',
 'repeat',
 'would',
 'upload',
 'note',
 'library',
 'refer']
'''
lda_discuss=['month',
 'reply',
 'assignment',
 'input',
 'print',
 'error',
 'question',
 'answer',
 'please',
 'output',
 'write',
 'ask',
 'number',
 'element',
 'edit',
 'convert',
 'array',
 'programming',
 'python',
 'integer',
 'list_a',
 'string',
 'value',
 'getting',
 'understand',
 'problem',
 'given',
 'function']
common=[]
common1=[]
labels={}
labels1={}
i=0
for each in lda_discuss:
    for each1 in lda_videos:
        if each==each1:
            common.append(names.index(each))
            common1.append(names1.index(each1))
            labels[names.index(each)]=names
            labels1[names1.index(each1)]=names.index(each)
            i=i+1
H2 = G.subgraph(common) #discuss
#H=nx.relabel_nodes(H,labels,False)
H3 = G1.subgraph(common1) #video  
print(H3.nodes())
H3=nx.relabel_nodes(H3,labels1)
print(H3.nodes())