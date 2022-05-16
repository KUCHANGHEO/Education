#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.getcwd()


# In[35]:


import pandas as pd
import numpy as np


# In[12]:


data = pd.read_csv('./gpascore.csv')
print(data)


# In[18]:


data.isnull().sum()


# In[20]:


data2 = data.dropna()
print(data2)


# In[22]:


print(data2['gre'])


# In[23]:


Y_data = data2['admit']
Y_data


# In[30]:


X_data = []


# In[32]:


for i, rows in data2.iterrows():
    X_data.append([rows['gre'], rows['gpa'], rows['rank']])
    
print(X_data)


# In[3]:


import tensorflow as tf


# # model정리

# In[6]:


model = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation = 'tanh'),
                            tf.keras.layers.Dense(128, activation = 'tanh'),
                            tf.keras.layers.Dense(1, activation = 'sigmoid')])


# In[8]:


model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')


# In[36]:


model.fit(np.array(X_data), np.array(Y_data), epochs = 1000 )


# In[38]:


p_data = ( [ [750, 3.70, 3], [400, 2.2, 1]])


# In[39]:


model.predict(p_data)


# In[ ]:




