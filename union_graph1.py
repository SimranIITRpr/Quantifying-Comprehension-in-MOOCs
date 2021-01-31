#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 15:44:58 2020

@author: simransetia
This program is used to construct merged network of video transcripts and QnA forum.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:42:22 2020

@author: simransetia
"""
import nltk
#f=open('/Users/simransetia/Documents/Dataset/video1.txt')
f=open('/Users/simransetia/Documents/Dataset/JOCWIKI1/discuss/discussion11.txt')
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
f=open('/Users/simransetia/Documents/Dataset/video11.txt')
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
lda_videos=['month',
 'check',
 'driver',
 'browser',
 'library',
 'particular',
 'thing',
 'input',
 'enter',
 'thirty',
 'using',
 'python',
 'number',
 'thousand',
 'message',
 'search',
 'valid',
 'selenium',
 'given',
 'click',
 'august',
 'chrome',
 'seventy',
 'nineteen',
 'prominent',
 'purpose',
 'procedure',
 'proceed',
 'proceeding',
 'process',
 'processing',
 'program',
 'programming',
 'probably',
 'import',
 'calendar']
lda_discuss=['question',
 'assignment',
 'answer',
 'webdriver',
 'option',
 'pattern',
 'perfect',
 'specific',
 'problem',
 'error',
 'browser']
common=[]
common1=[]
labels={}
labels1={}
i=0
for each in lda_discuss:
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
Hud=nx.Graph()
Hud.add_edges_from(list(H1.edges())+list(H.edges()))
Hud.remove_nodes_from(['press','ask'])
Hud.add_nodes_from(list(H1.nodes(data=True))+list(H.nodes(data=True))) 