#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 23:22:03 2020

@author: simransetia
This program is used to construct merged network of video transcripts and wiki articles.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 19:56:18 2020

@author: simransetia

"""
import nltk
import matplotlib.pyplot as plt
from w2vec import Gw2v
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer, SnowballStemmer
f=open('/Users/simransetia/Documents/Dataset/JOCWIKI1/doc1/week8.txt')
corpus=f.readlines()


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        #print("word"+word)
        return word
    else:
        #print("lemma"+lemma)
        return lemma
    
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)
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
        tokens = [get_lemma(token) for token in tokens]
        tokens = [get_lemma2(token) for token in tokens]
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
f=open('/Users/simransetia/Documents/Dataset/video8.txt')
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
        tokens = [get_lemma(token) for token in tokens]
        tokens = [get_lemma2(token) for token in tokens]
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
lda_article=['tuple',
 'index',
 'using',
 'sentiment',
 'install',
 'python',
 'image',
 'tuples',
 'available',
 'pointer',
 'procedure',
 'problem',
 'printing',
 'print',
 'prediction',
 'string',
 'element',
 'account',
 'anagram',
 'create',
 'positive',
 'player']
lda_videos=['count',
 'value',
 'anagram',
 'write',
 'tuple',
 'image',
 'string',
 'print',
 'technique',
 'sentiment',
 'excel',
 'using',
 'positive',
 'example',
 'number',
 'equal',
 'thirty',
 'analysis',
 'happen',
 'sort',
 'ascii',
 'check',
 'information',
 'time',
 'seven',
 'store',
 'download',
 'screen',
 'program']

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

H = G.subgraph(common)
H=nx.relabel_nodes(H,labels)
#H=nx.relabel_nodes(H,labels,False)
H1 = G1.subgraph(common1)  
H1=nx.relabel_nodes(H1,labels1)
Hu=nx.Graph()
Hu.add_edges_from(list(H1.edges())+list(H.edges()))
Hu.add_nodes_from(list(H1.nodes(data=True))+list(H.nodes(data=True))) 
Hu.remove_nodes_from(['positioning', 'press'])
Huw=nx.Graph()
Huw.add_edges_from(list(H1.edges())+list(Gw2v.edges()))
Huw.add_nodes_from(list(H1.nodes(data=True))+list(Gw2v.nodes(data=True)))
nx.write_gml(H,"H1.gml")
nx.write_gml(H1,"H11.gml")
nx.write_gml(Gw2v,"w2v1.gml")
nx.write_gml(Hu,"Hu1.gml")
nx.write_gml(Huw,"Huw1.gml")