#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 00:27:43 2020

@author: simransetia
This program is used to build merged network of word2vec and video transcripts.
"""

from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
  
warnings.filterwarnings(action = 'ignore') 
  
import gensim 
from gensim.models import Word2Vec 
import networkx as nx
  
#  Reads ‘alice.txt’ file 
sample = open('/Users/simransetia/Documents/Dataset/video3.txt')
s = sample.read() 
  
# Replaces escape character with space 
f = s.replace("\n", " ") 
  
data = [] 
  
# iterate through each sentence in the file 
for i in sent_tokenize(f): 
    temp = [] 
      
    # tokenize the sentence into words 
    for j in word_tokenize(i): 
        temp.append(j.lower()) 
  
    data.append(temp) 
  
# Create CBOW model 
model1 = gensim.models.Word2Vec(data, min_count = 1,  
                              size = 100, window = 5) 
  
# Print results 
#print(model1.similarity('programming', 'right')) 
      
lda_videos=['value',
 'number',
 'print',
 'write',
 'player',
 'point',
 'estimate',
 'thing',
 'random',
 'computer',
 'particular',
 'order',
 'index',
 'equal',
 'people',
 'seven',
 'functionality',
 'start',
 'change',
 'generate',
 'percent',
 'given',
 'example',
 'shopping',
 'function',
 'right',
 'check',
 'default',
 'variable',
 'small',
 'fizzbuzz',
 'using',
 'different',
 'person',
 'create',
 'item',
 'jumble',
 'word',
 'basically',
 'question',
 'answer',
 'seventy',
 'length',
 'element',
 'python',
 'minus',
 'programming',
 'count',
 'really']
dictsim={}
Gw2v=nx.Graph()
Gw2v.add_nodes_from(lda_videos,labels=lda_videos)
for each in lda_videos:
    for each1 in lda_videos:
        if each!=each1:
            if(model1.similarity(each,each1)>0.50 and Gw2v.has_edge(each,each1)==False):
                Gw2v.add_edge(each,each1)
    