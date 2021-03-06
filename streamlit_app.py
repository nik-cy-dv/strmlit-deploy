#from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import pickle
import re
import nltk
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.corpus import wordnet

from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

# load the model from disk
path = 'model2_gv_deepl.h5'
filename = 'tokenizer.pkl'
tokenizer = pickle.load(open(filename, 'rb'))

st.title('Sexual Harassment Classification')
st.image('https://www.democracyandme.org/wp-content/uploads/2019/10/Me-Too-Image-2.jpeg', width = 475)
text = st.text_input('Enter the event description:')
if text == None or text == '':
  st.markdown('**Enter a text in '' to get result...**')

else:
  def preprocess(text):
    
    """performs common expansion of english words, preforms preprocessing"""
  
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"won\'t", "will not", text)   # decontracting the words
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)

    text = re.sub(r'\w+:\s?','',text)                                            ## removing anyword:
    text = re.sub('[([].*?[\)]', '', text)                                       ## removing sq bracket and its content
    text = re.sub('[<[].*?[\>]', '', text)                                       ## removing <> and its content
    text = re.sub('[{[].*?[\}]', '', text)                                       ## removing {} and its content
    
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])       ## lemmatizing the word

    text = re.sub(r'\W', ' ', str(text))                                         # Remove all the special characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)                                  # remove all single characters 
    text = re.sub(r"[^A-Za-z0-9]", " ", text)                                    # replace all the words except "A-Za-z0-9" with space  
    text = re.sub(r'[^\w\s]','',text)
    text = ' '.join(e for e in text.split() if e.lower() not in set(stopwords.words('english')) and len(e)>2)  
    # convert to lower and remove stopwords discard words whose len < 2
    
    text = re.sub("\s\s+" , " ", text)                                           ## remove extra white space  lst
    text = text.lower().strip()   

    return text
  
  st.markdown("""<style>.big-font</style>""", unsafe_allow_html=True)
  st.markdown('<p class="big-font">Possible Act:</p>', unsafe_allow_html=True)
	
  #def end_to_end_pipeline(string):
  path = 'model2_gv_deepl.h5'
  result = []
  x = preprocess(text)
  sent_token = tokenizer.texts_to_sequences([x])
  sent_token_padd = pad_sequences(sent_token, maxlen=300, dtype='int32', padding='post', truncating='post')
  model = tf.keras.models.load_model(path)
  pred = model.predict(sent_token_padd)

  row, column = pred.shape
  predict = np.zeros((row, column))
  for i in range(row):
    for j in range(column):
      if pred[i,j]>0.5:
        predict[i,j] = 1
          

    #if request.method == 'POST':
  for k in range(predict.shape[0]):
    if predict[k][0] == 1.0:
      result.append('commenting')
    if predict[k][1] == 1.0:
      result.append('ogling')
    if predict[k][2] == 1.0:
      result.append('groping')
    if np.sum(predict) == 0.0:
      result.append('None')
    
  print(f'possible action : {result}')
  st.markdown(result)

