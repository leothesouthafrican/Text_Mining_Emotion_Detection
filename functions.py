import pandas as pd
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup
from tqdm import tqdm_notebook as tqdm
import re


def txt_to_df(list_of_sentences, drop_first_row = True):

    #Creating dataframe for storing training examples
    train_df = pd.DataFrame(columns=('sentence', 'emotion'))
    i = 0  
    sentence = "" 
    emotion = ""

    #Iterating over sentences and separating them into specific df columns
    for line in list_of_sentences:
        sentence = line.split("\t")[0]
        emotion = line.split("\t")[1]
        train_df.loc[i] = [sentence, emotion]
        i =i+1

    #dropping first line  if necessary   
    if drop_first_row:
        train_df = train_df[1:]

    return train_df

def label_counter(dataframe, field, set_name):
    fig = px.histogram(dataframe, x=field, title = "Distribution of Emotions in " + set_name + " Set")
    fig.show()

def word_counter(dataframe_column, set_name, number_words = 25):
    
    #Joining words in dataframe_column
    words_in_df = ' '.join(dataframe_column).split()
    
    # Counting frequency of words in dataframe_column
    freq = pd.Series(words_in_df).value_counts()

    #Creating a datframe of the count
    temp_df = pd.DataFrame(freq)[:number_words]

    #Resetting index and renaming columns
    temp_df.reset_index(inplace=True)
    temp_df.rename(columns = {"index":"word", 0:"count"}, inplace = True)

    #Outputting histogram
    fig = px.histogram(temp_df,x = "word", y = "count", title = "Distribution of " + str(number_words) + " Most Frequent Words in " + set_name + " Set")
    fig.show()

def clean(text_list, lemmatize, stemmer):
    """
    Function that a receives a list of strings and preprocesses it.
    
    :param text_list: List of strings.
    :param lemmatize: Tag to apply lemmatization if True.
    :param stemmer: Tag to apply the stemmer if True.
    """
    lemma = WordNetLemmatizer()
    snowball_stemmer = SnowballStemmer('english')

    updates = []
    for j in tqdm(range(len(text_list))):
        
        text = text_list[j]
        
        #LOWERCASE TEXT
        text = text.lower()
        
        #REMOVE NUMERICAL DATA AND PUNCTUATION
        text = re.sub("[^a-zA-Z]", ' ', text)
        
        #REMOVE TAGS
        text = BeautifulSoup(text).get_text()
        
        if lemmatize:
            text = " ".join(lemma.lemmatize(word) for word in text.split())
        
        if stemmer:
            text = " ".join(snowball_stemmer.stem(word) for word in text.split())
        
        updates.append(text)
        
    return updates

def update_df(dataframe, list_updated):
    dataframe.update(pd.DataFrame({"Text": list_updated}))