#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.getcwd()


# In[2]:


import pandas as pd


# In[3]:


# Raw Data Loading
mnist = pd.read_csv('./mnist_train.csv')


# In[4]:


mnist.head()


# In[5]:


mnist.describe()


# In[6]:


y_data= mnist['label']


# In[7]:


y_data.head()


# In[8]:


X_data = mnist.drop('label', axis = 1, inplace=False)


# In[9]:


X_data.head()


# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[17]:


scaler = MinMaxScaler()
scaler.fit(X_data)
X_norm_data = scaler.transform(X_data)


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X_norm_data, y_data, test_size=0.2)


# In[19]:


X_train.shape


# In[14]:


# model 설정


# In[15]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD


# In[20]:


model = Sequential()


# In[21]:


model.add(Flatten(input_shape=(784,))) # input Layer
model.add(Dense(units=10, activation= 'softmax'))


# In[22]:


model.compile(optimizer = SGD(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[33]:


# 5. 모델 훈련
# verbose: Integer. 0, 1, or 2. 
# Verbosity mode. 
# 0 = silent, 
# 1 = progress bar, 
# 2 = one line per epoch.


# In[37]:


history = model.fit(X_train,y_train,
              validation_data=(X_test,y_test),
              steps_per_epoch= 50,
              validation_steps=50,
              epochs=50, verbose=1)


# In[24]:


model.evaluate(X_test , y_test)


# In[34]:


# 6. 모델 저장
model.save('model3.h5')


# In[25]:


model.summary()


# In[29]:


import matplotlib.pyplot as plt


# In[30]:


acc = history.history['accuracy']
loss= history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']


# In[35]:


# acc


# In[31]:


epochs = range(len(acc))


# In[32]:


plt.plot(epochs, acc, 'bo', label= 'Training accuracy')
plt.plot(epochs, val_acc, 'b', label= 'Training accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label= 'Training loss')
plt.plot(epochs, val_loss, 'b', label= 'Training loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:




