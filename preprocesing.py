import string
import re

import nltk
from nltk.corpus import stopwords

import numpy as np 

stopword = stopwords.words('english')
wn = nltk.WordNetLemmatizer()
ps = nltk.PorterStemmer()


def remove_punctions(text):
    text_nopunc = "".join([char for char in text if char not in string.punctuation])
    return text_nopunc

def tokenize(text):
    tokenize = re.split('\W+',text)
    return tokenize

def rem_stopword(tokenized_text):
    
    removed_stopword=[char for char in tokenized_text if char not in stopword]
    return removed_stopword

def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text

def bag_of_words(word,total_words):
    bag = np.zeros(len(total_words), dtype=np.float32)
    for idx, w in enumerate(total_words):
        if w in word: 
            bag[idx] = 1

    return bag

if __name__ =="__main__":
    a= "I've been searching for the right words, to Thank you!!!"
    print(a)
    a_nopunc = remove_punctions(a)
    print(a_nopunc)
    a_tok = tokenize(a_nopunc)
    print(a_tok)
    a_stop = rem_stopword(a_tok)
    print(a_stop)
    a_lem = lemmatizing(a_stop)
    print(a_lem)
    a_stem = stemming(a_stop)
    print(a_stem)

    bag = bag_of_words(a_stem)
    print(bag)
