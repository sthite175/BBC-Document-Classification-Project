import os
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from contractions import fix
from unidecode import unidecode
from  nltk.stem import WordNetLemmatizer

# Stopwords List
stopwords_list = stopwords.words("english")

def read_extract_text_file(path):
    with open(path,'r') as file:
        text_data = file.read()
    
    return text_data


def preprocess_data(text):
    text = text.lower()
    text = text.replace("\n"," ").replace("\t"," ")
    text = re.sub("\s+"," ",text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    
    # tokens
    tokens = word_tokenize(text)
    
    data = [i for i in tokens if i not in punctuation]
    data = [i for i in data if i not in stopwords_list]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    final_text = []
    for i in data:
        word = lemmatizer.lemmatize(i)
        final_text.append(word)
        
    return " ".join(final_text)


# <textarea name="text" rows="1" cols="5"></textarea>


# <center>
#     <h3>BBC Document Classification Model</h3>
#     <form method="POST" action="/prediction" enctype="multipart/form-data">
#         <label for="file">Upload Document File:</label>
#         <br>
#         <input type="file" name="file" id="file" accept=".txt" required>
#         <br>
#         <br>
#         <input type="submit" value="Classify Document">
#         <br>
#         <br>
#         <p style="color: blue;">
#             {{result}}
#         </p>
#     </form>
#     </center>
