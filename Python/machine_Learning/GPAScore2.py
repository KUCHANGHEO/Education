#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np


# In[17]:


data = pd.read_csv('./gpascore.csv')
print(data)


# # 결측데이터 전처리

# In[18]:


data.isnull().sum()


# In[19]:


data2 = data.dropna()
print(data2)


# In[20]:


Y_data = data2['admit']
Y_data


# In[22]:


X_data = data2.drop(['admit'], axis = 1)
print(X_data)


# In[2]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[11]:


del model


# In[12]:


model = Sequential()


# In[13]:


model.add(Dense(64, input_dim = 3, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))


# In[14]:


model.summary()


# In[15]:


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')


# In[23]:


model.fit(np.array(X_data), np.array(Y_data), epochs = 10 , batch_size = 10)


# In[25]:


import cv2


# In[27]:


cv2.__version__


# In[ ]:




