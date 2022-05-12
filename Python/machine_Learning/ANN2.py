#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.getcwd()


# In[1]:


import pandas as pd


# In[6]:


# Raw Data Loading
mnist = pd.read_csv('./mnist_train.csv')


# In[7]:


mnist.head()


# In[9]:


mnist.describe()


# In[10]:


y_data= mnist['label']


# In[11]:


y_data.head()


# In[14]:


X_data = mnist.drop('label', axis = 1, inplace=False)


# In[15]:


X_data.head()


# In[21]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[23]:


scaler = MinMaxScaler()
scaler.fit(X_data)
X_norm_data = scaler.transform(X_data)


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X_norm_data, y_data, test_size=0.2)


# In[41]:


X_train.shape


# In[42]:


# model 설정


# In[37]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD


# In[28]:


model = Sequential()


# In[34]:


model.add(Flatten(input_shape=(784,))) # input Layer
model.add(Dense(units=10, activation= 'softmax'))


# In[38]:


model.compile(optimizer = SGD(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[43]:


model.fit(X_train,y_train, epochs=100, verbose=1, validation_split=0.2)


# In[45]:


model.evaluate(X_test , y_test)


# In[47]:


model.summary()


# In[ ]:




