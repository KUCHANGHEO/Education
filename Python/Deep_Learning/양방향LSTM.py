#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# In[4]:


word_num = 10000
batch_size = 128
max_len = 200
embedding_len = 100


# In[6]:


(x_train, y_train), (x_test,y_test) = imdb.load_data(num_words = word_num)


# In[8]:


pad_x_train = sequence.pad_sequences(x_train, maxlen= max_len)
pad_x_test = sequence.pad_sequences(x_test, maxlen= max_len)


# In[9]:


y_train = np.array(y_train)
y_test = np.array(y_test)


# In[10]:


model = Sequential()
model.add(Embedding(word_num, 128, input_length=max_len))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))


# In[11]:


model.compile(loss="binary_crossentropy", optimizer= "adam", metrics=['accuracy'])


# In[ ]:


model.fit(pad_x_train,y_train,
          batch_size= batch_size,
          epochs=4,
          validation_data=[pad_x_test,y_test])


# In[ ]:


model.summary()


# In[ ]:


print("훈련 데이터셋 평가...")
(loss, accuracy) = model.evaluate(pad_x_train, y_train, batch_size = 384, verbose=1)
print('Training accuracy', model.metrics_names, acc)
print('Training loss', model.metrics_names, loss)
print("테스트 데이터셋 평가...")
(loss, accuracy) = model.evaluate(pad_x_test, y_test, batch_size = 384, verbose=1)
print('Testing accuracy', model.metrics_names, acc)
print('Testing loss', model.metrics_names, loss)


# In[ ]:




