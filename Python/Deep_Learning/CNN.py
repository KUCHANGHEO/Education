#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('mnist_train.csv')


# In[3]:


df.head()


# In[4]:


import numpy as np
import tensorflow as tf              # tensorflow 기본
from tensorflow.keras.models import Sequential # 모델 Box 정의
from tensorflow.keras.layers import Flatten, Dense # 모델 BOX의 Input Layer와 Output Layer

from tensorflow.keras.optimizers import SGD # 알고리즘 담당
from sklearn.model_selection import train_test_split # train, test 데이터를 분리
from sklearn.preprocessing import MinMaxScaler # 데이터 정규화 - 큰 숫자를 작은 숫자로 변경 (-1 ~ + 1)


# In[5]:


X_data = df.drop('label',axis=1)


# In[6]:


X_data.head()


# In[7]:


y_data = df['label']


# In[8]:


y_data.head()


# In[9]:


# 픽셀 데이터를 정규화(0~1 사이의 실수로 변환)
scaler = MinMaxScaler()                    # scaler 객체 생성
scaler.fit(X_data)                         # scaler 객체를 학습
normal_x_data = scaler.transform(X_data)   # scaler를 통해서 실제값을 변환


# In[10]:


norm_train_x_data, norm_test_x_data, train_y_data, test_y_data = train_test_split(normal_x_data,y_data,test_size=0.3)


# In[11]:


norm_train_x_data = norm_test_x_data.reshape(-1,28,28,1)
norm_test_x_data = norm_test_x_data.reshape(-1,28,28,1)


# In[12]:


from tensorflow.keras.layers import Conv2D, MaxPool2D


# In[13]:


model = Sequential()


# In[14]:


# Convolution
model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    strides=(1,1),
    activation='relu',
    input_shape=(28,28,1)))


# In[15]:


model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    strides=(1,1),
    activation='relu'))


# In[16]:


model.add(MaxPool2D(pool_size=(2,2)))


# In[17]:


model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    strides=(1,1),
    activation='relu'))


# In[18]:


# FC layer(DNN) 의: input layer
model.add(Flatten(input_shape=())) # 전체 4차원 에서 2차원으로 바꿔주는것


# In[19]:


# hidden Layer
model.add(Dense(units=256, activation='relu'))


# In[20]:


# output layer
model.add(Dense(units=10, activation='softmax'))


# In[21]:


# model이 어떻게 동작하는지를 지정
model.compile(optimizer=SGD(learning_rate=0.0001),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[34]:


# 설정을 모두 마치었으면 모델 학습
history = model.fit(norm_train_x_data,train_y_data, epochs = 30,
             verbose= 1,
             validation_split=0.2)


# In[35]:


model.summary()


# In[36]:


model.evaluate(norm_test_x_data, test_y_data)


# In[37]:


import matplotlib.pyplot as plt


# In[38]:


acc = history.history['accuracy']
loss= history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']


# In[39]:


epochs = range(len(acc))


# In[40]:


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




