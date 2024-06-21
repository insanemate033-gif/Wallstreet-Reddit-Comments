#!/usr/bin/env python
# coding: utf-8

# In[1]:


# quick analysis of WallstreetBets redit comments


# In[2]:


# loading in all the essentials
import pandas as pd
import numpy as np 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk import ngrams
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


# reading in the dataset
df = pd.read_csv('reddit_wsb.csv')
df.dtypes


# In[65]:


import sys
print(sys.executable)


# In[66]:


get_ipython().system('{sys.executable} -m pip install emoji')


# In[68]:


get_ipython().system('{sys.executable} -m pip install --upgrade typing_extensions')


# In[69]:


import emoji
print(emoji.emojize("Python is fun :thumbs_up:"))


# In[ ]:





# In[4]:


#check the info column
df.info()


# In[5]:


#create a timestamp
df.timestamp = pd.to_datetime(df.timestamp)


# In[6]:


#create a day, hour, dayofweek column and check the head of the data
df['day'] =df['timestamp'].dt.date
df['day'] =pd.to_datetime(df['day'])
df['hour'] = df.timestamp.dt.hour
df['dayofweek'] =df.timestamp.dt.day_name()
df.head()


# In[7]:


#create a new data frame that isolate only march check the amount of comments
df2 = df[df['timestamp']>='2021-02-01']
print(df.shape)
print(df2.shape)


# In[8]:


plt.style.use('ggplot')
df2.groupby('day')['title'].count().plot(figsize=(16,4))
plt.title("The trend of comments and the max comment count was " + str(df2.groupby('day')['title'].count().max()));


# In[10]:


import nltk

# Ensure 'punkt' is downloaded
nltk.download('punkt')

from nltk.tokenize import word_tokenize

text = "Hello, world! This is a test sentence."
tokens = word_tokenize(text)
print(tokens)


# In[11]:


# let joins all the title rows and tokenize them into words
comments =" ".join(df['title'])
words =word_tokenize(comments)


# In[12]:


#what are the five top days for comments
df2.groupby('day',as_index=False)['title'].count().sort_values(by='title',ascending=False).head()


# In[13]:


comments


# In[15]:


words


# In[23]:


import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[27]:


#lets creat a small function to clean the words of punctuations, stop words and lemmatize the words
def clean_words(new_tokens):
    new_tokens = [t.lower() for t in new_tokens]
    new_tokens =[t for t in new_tokens if t not in stopwords.words('english')]
    new_tokens = [t for t in new_tokens if t.isalpha()]
    lemmatizer = WordNetLemmatizer()
    new_tokens =[lemmatizer.lemmatize(t) for t in new_tokens]
    return new_tokens


# In[28]:


lowered = clean_words(words)


# In[29]:


bow = Counter(lowered)


# In[30]:


bow


# In[31]:


data = pd.DataFrame(bow.items(),columns=['word','frequency']).sort_values(by='frequency',ascending=False)


# In[32]:


data =data.head(20)


# In[33]:


data


# In[34]:


sns.barplot(x='frequency',y='word',data=data)


# In[35]:


bow2 =Counter(ngrams(lowered,2))


# In[36]:


bow2


# In[52]:


def word_frequency(sentence):
    sentence =" ".join(sentence)
    new_tokens = word_tokenize(sentence)
    new_tokens = [t.lower() for t in new_tokens]
    new_tokens =[t for t in new_tokens if t not in stopwords.words('english')]
    new_tokens = [t for t in new_tokens if t.isalpha()]
    lemmatizer = WordNetLemmatizer()
    new_tokens =[lemmatizer.lemmatize(t) for t in new_tokens]
    counted = Counter(new_tokens)
    counted_2= Counter(ngrams(new_tokens,2))
    counted_3= Counter(ngrams(new_tokens,3))
    word_freq = pd.DataFrame(counted.items(),columns=['word','frequency']).sort_values(by='frequency',ascending=False)
    word_pairs =pd.DataFrame(counted_2.items(),columns=['pairs','frequency']).sort_values(by='frequency',ascending=False)
    trigrams =pd.DataFrame(counted_3.items(),columns=['trigrams','frequency']).sort_values(by='frequency',ascending=False)
    return word_freq,word_pairs,trigrams
    


# In[56]:


data2,data3,data4 = word_frequency(df['title'])


# In[57]:


data4


# In[58]:


fig, axes = plt.subplots(3,1,figsize=(8,20))
sns.barplot(ax=axes[0],x='frequency',y='word',data=data2.head(30))
sns.barplot(ax=axes[1],x='frequency',y='pairs',data=data3.head(30))
sns.barplot(ax=axes[2],x='frequency',y='trigrams',data=data4.head(30))


# In[ ]:




