#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')




data = pd.read_csv('./data/medicalCassificationPart2-add.csv',sep=";", encoding='cp1250')
data_copi = data.copy()


# In[8]:




# In[4]:


# laczy ciągi w seri 
def word_in_col(data, col_name):
    skin = data[col_name].str.cat(sep=' ')
    
    tokens = word_tokenize(skin)
    
    vocabulary = set(tokens)
    print(len(tokens))
    
    frequency_dist = nltk.FreqDist(tokens)
    
    sorted_list = sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)
    print(sorted_list)
    
    stop_words = set(stopwords.words('english'))
    
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [w for w in tokens if not w in {'xxx',','}]
    frequency_dist = nltk.FreqDist(tokens)
    print('ilość słów: ',len(tokens))
    sorted_list = sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)
    print(sorted_list)
    print()
    print(' ilość po sortowaniu: ',len(sorted_list))
    
    porter = PorterStemmer()
    stems = []
    for t in tokens:    
        stems.append(porter.stem(t))
    print()
    return stems

sorted_list = word_in_col(data,'medical history')
def frequency_dist(stems):
    frequency_dist = nltk.FreqDist(stems)
    sorted_list = sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)
    print(sorted_list)
    print('po sort:',len(sorted_list))
    return frequency_dist,sorted_list
frequency_dist,sorted_list = frequency_dist(sorted_list)


# In[296]:


frequency_dist


# In[9]:


tokens = word_tokenize(skin)
vocabulary = set(tokens)


# In[189]:


vocabulary = set(tokens)


# In[190]:


# wykozystano biblioteke nlkt do wstepnej obrobki tekstu
#podzielenie tekstu na tokey slow

print(len(vocabulary))
frequency_dist = nltk.FreqDist(tokens)
sorted_list = sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50]
print(sorted_list)


# In[5]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]
tokens = [w for w in tokens if not w in {'xxx',','}]


# In[6]:


frequency_dist = nltk.FreqDist(tokens)

sorted_list = sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)
print(sorted_list)


# In[7]:


#NLTK provides several stemmer interfaces like Porter stemmer, #Lancaster Stemmer, Snowball Stemmer
from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()
stems = []
for t in tokens:    
    stems.append(porter.stem(t))
print(len(stems))


# In[196]:


frequency_dist = nltk.FreqDist(stems)
sorted_list = sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)
print(sorted_list)


# In[183]:


frequency_dist = nltk.FreqDist(tokens)
t= sorted(frequency_dist,key=frequency_dist.__getitem__, reverse=True)[0:50]
print(t)


# In[352]:


def graph():
    frequency_dist.plot(41,cumulative=False)


# In[400]:


graph()


# In[ ]:





# In[ ]:





# In[357]:


from nltk.corpus import wordnet
from itertools import chain

def get_synonyms(df, column_name, N):
    L = []
    
    for i in sorted_list:
        syn = wordnet.synsets(i)
        #flatten all lists by chain, remove duplicates by set
        lemmas = list(set(chain.from_iterable([w.lemma_names() for w in syn])))
        for j in lemmas[:N]:
            #append to final list
            L.append([i, j])
    #create DataFrame
    return (pd.DataFrame(L, columns=['word','syn']))    

#add number of filtered synonyms

df1 = get_synonyms(data, 'skin', 10)
print (df1.head(50))


# In[217]:


for x in sorted_list:
  


# In[394]:


def convert_lower_case(data):
    return np.char.lower(data)
def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words:
            new_text = new_text + " " + w
    return new_text
def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data
def remove_apostrophe(data):
    return np.char.replace(data, "'", "")
def stemming(data):
    stemmer= PorterStemmer()
    
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text


# In[ ]:





# In[385]:


from  sklearn.feature_extraction.text  import  CountVectorizer 


# In[ ]:





# In[416]:


skin = data['main symptoms'].str.cat(sep=' ')
tokens = word_tokenize(skin)


# In[417]:


from nltk.collocations import *
tokens = [w for w in tokens if not w in {'xxx',',','and','are','is','the'}]
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(tokens)
scores = finder.score_ngrams( bigram_measures.raw_freq )
print(scores)
finder.nbest (bigram_measures.pmi, 10) # doctest: + NORMALIZE_WHITESPACE


# In[ ]:




