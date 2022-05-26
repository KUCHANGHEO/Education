#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download()


# In[2]:


text = nltk.word_tokenize("Is it possible distinguishing cats and dogs")
text


# In[4]:


nltk.download('averaged_perceptron_tagger')


# In[5]:


nltk.pos_tag(text)


# In[6]:


nltk.download('punkt')


# In[7]:


string1='my favorite subject is math'
string2='my favorite subject is math, english, economic and computer science'


# In[8]:


nltk.word_tokenize(string1)


# In[9]:


nltk.word_tokenize(string2)


# In[1]:


from konlpy.tag import Komoran


# In[2]:


komoran = Komoran()


# In[4]:


print(komoran.morphs('딥러닝 수업이 쉽나요? 어렵나요?'))


# In[5]:


print(komoran.pos('소파 위에 있는 동물이 고양이인가요? 강아지인가요?'))


# In[6]:


import csv
from konlpy.tag import Okt


# In[13]:


f = open(r'./data/ratings_train.txt', 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
rdw = list(rdr)
f.close()


# In[14]:


twitter = Okt()


# In[15]:


result = []
for line in rdw:
    malist = twitter.pos( line[1], norm=True, stem=True)
    r = []
    for word in malist:
        if not word[1] in ["Josa","Eomi","Punctuation"]:
            r.append(word[0])
    rl = (" ".join(r)).strip()
    result.append(rl)
    print(rl)


# In[ ]:


with open("NaverMovie.nlp",'w', encoding='utf-8') as fp:
    fp.write("\n".join(result))


# In[10]:


from nltk import sent_tokenize
text_sample = 'Natural Language Processing, or NLP, is the process of extracting the meaning, or intent, behind human language, in the'
tokenized_sentences = sent_tokenize(text_sample)
print(tokenized_sentences)


# In[11]:


from nltk import word_tokenize
sentence = 'This book is for deep learning learners'
words = word_tokenize(sentence)
print(words)


# In[17]:


from tensorflow.keras.preprocessing.text import text_to_word_sequence
sentence = "it’s nothing that you don’t already know except most people aren’t aware of how their inner world works."
words = text_to_word_sequence(sentence)
print(words)


# In[18]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
sample_text = "One of the first things that we ask ourselves is what are the pros and cons of any task we perform."
text_tokens = word_tokenize(sample_text)
tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]
print("불용어 제거 미적용:", text_tokens, '\n')
print("불용어 제거 적용:",tokens_without_sw)


# In[19]:


#포터 알고리즘
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem('obesses'),stemmer.stem('obssesed'))
print(stemmer.stem('standardizes'),stemmer.stem('standardization'))
print(stemmer.stem('national'), stemmer.stem('nation'))
print(stemmer.stem('absentness'), stemmer.stem('absently'))
print(stemmer.stem('tribalical'), stemmer.stem('tribalicalized'))


# In[1]:


#포터 알고리즘
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()
print(stemmer.stem('obesses'),stemmer.stem('obssesed'))
print(stemmer.stem('standardizes'),stemmer.stem('standardization'))
print(stemmer.stem('national'), stemmer.stem('nation'))
print(stemmer.stem('absentness'), stemmer.stem('absently'))
print(stemmer.stem('tribalical'), stemmer.stem('tribalicalized'))


# In[2]:


#표제어 추출(Lemmatization)
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer()
print(stemmer.stem('obsesses'),stemmer.stem('obsessed'))
print(lemma.lemmatize('standardizes'),lemma.lemmatize('standardization'))
print(lemma.lemmatize('national'), lemma.lemmatize('nation'))
print(lemma.lemmatize('absentness'), lemma.lemmatize('absently'))
print(lemma.lemmatize('tribalical'), lemma.lemmatize('tribalicalized'))

