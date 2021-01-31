#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 22:01:48 2020

@author: simransetia
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:07:57 2018

@author: simransetia
This program is used to figure out the important keywords in the underlying text using LDA topic modelling
"""


import matplotlib.pyplot as plt
import spacy
import random
import gensim
import pickle
from spacy.lang.en import English
from gensim import corpora
from gensim.corpora import Dictionary
parser = English()
#from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import nltk
spacy.load('en')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import wordnet as wn
en_stop = set(nltk.corpus.stopwords.words('english'))
spacy.load('en')
from spacy.lang.en import English
import xml.etree.cElementTree as ec
parser = English()
i=0
text=''
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens
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
    #return stemmer.stem(WordNetLemmatizer().lemmatize(word, pos='v'))
# Function Call For Connecting To Our Database 

def prepare_text_for_lda(text):
    tokens1=[]
    tokens2=[]
    tokens = tokenize(text)
    for each in tokens:
        
        #print(each)
        if '\\n' in each:
            tokenss=str(each)
            tokenss=tokenss.replace('\\n','')
            #tokenss=list(tokenss)
            
            tokens1.append(tokenss)
        #elif each=='to' or each=='the' or each=='of' or each=='.' or each=="\\" :
            
            #tokens.remove(each)
        else:
            tokens1.append(each)
    tokens2 = [token for token in tokens1 if len(token) > 4]
    tokens3 = [token for token in tokens2 if token not in en_stop]
    tokens4 = [get_lemma(token) for token in tokens3]
    tokens5 = [get_lemma2(token) for token in tokens4]

    return tokens5
def topic_model(text):
    tokens=prepare_text_for_lda(text)
    text_data=[]
    text_data.append(tokens)
    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]
        
    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')
    NUM_TOPICS = 1
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=20)
    return topics
text1=[]
rev=[]
f1=open("article8.txt","w")
#f1=open("video11.txt","w")
#f1=open("discuss11.txt","w")
f=open('/Users/simransetia/Documents/Dataset/JOCWIKI1/doc1/week8.txt')
#f=open('/Users/simransetia/Documents/Dataset/video11.txt')
#f=open('/Users/simransetia/Documents/Dataset/JOCWIKI1/discuss/discussion11.txt')
text1=f.readlines()
textdata=[]
for each in text1:
    if each!='' and 'comments' not in each and '(talkcontribs)' not in each:
        textdata.append(each)
        
tokens=prepare_text_for_lda(str(textdata))
                        #print(tokens)
text_data=[]
text_data.append(tokens)
dictionary = corpora.Dictionary(text_data)
                       # dictionary.filter_extremes(no_below=1, no_above=0)
corpus = [dictionary.doc2bow(text) for text in text_data]
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')
NUM_TOPICS = 10
                        #print(corpus)
if corpus:
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=100)
    ldamodel.save('model5.gensim')
    topics = ldamodel.print_topics(num_words=10)
    for each in topics:
        print(each)
        topic=str(each)
        f1.write("\n"+topic+"\n")
                   

f1.close()
import re
set2d=[]
with open('/Users/simransetia/Documents/simran/Laboratory/Wiki Education/article8.txt', 'r')  as f:
    lines=f.readlines()
    for each in lines:
        topic_lines=[]
        if '(' in each:
            topic_lines.append(each)
            #print(topic_lines)
            topic_lines=str(topic_lines)
            #print(topic_lines)
            searchobj=re.findall(r'\d\.\d+\*\"\w+\"',topic_lines)
            for each in searchobj:
                
                
                search_=str(each)
                search_topic=re.findall(r'\"\w+\"',search_)
                search_topic=str(search_topic)
                search_topic=search_topic[3:len(each)-5]
                #print(search_topic)
                if search_topic not in set2d:
                    set2d.append(search_topic)
                