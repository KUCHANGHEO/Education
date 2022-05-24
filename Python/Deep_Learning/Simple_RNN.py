#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np
from tensorflow import keras


# # 순환 신경망(Recurrent Neural Network, RNN)
# 
# - 루프(loop)를 가진 신경망의 한 종류
# 
# - 시퀸스의 원소를 순회하면서 지금까지 처리한 정보를 상태(state)에 저장

# # 순환 신경망 레이어 (RNN Layer)
# 
# - 입력: (timesteps, input_features)
# 
# - 출력: (timesteps, output_features)

# In[1]:


import numpy as np


# In[2]:


timesteps = 100
input_features = 32
output_features = 64


# In[3]:


inputs = np.random.random((timesteps, input_features))
inputs


# In[4]:


state_t = np.zeros((output_features))
state_t


# In[7]:


W = np.random.random((output_features, input_features))  # ((64, 32))
U = np.random.random((output_features, output_features)) # ((64,64))
b = np.random.random((output_features))                  # ((64))


# In[16]:


W


# In[17]:


U


# In[18]:


b


# In[8]:


sucessive_outputs = []


# In[10]:


for input_t in inputs:
    output_t =  np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    sucessive_outputs.append(output_t)
    state_t = output_t

final_output_sequence = np.stack(sucessive_outputs, axis = 0)


# In[14]:


type(sucessive_outputs)
sucessive_outputs


# In[15]:


type(final_output_sequence)
final_output_sequence


# In[34]:


from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential


# In[22]:


model = Sequential()


# In[23]:


model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.summary()


# In[24]:


model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences= True))
model.summary()


# In[25]:


model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences= True))
model.add(SimpleRNN(32, return_sequences= True))
model.add(SimpleRNN(32, return_sequences= True))
model.add(SimpleRNN(32))
model.summary()


# In[26]:


from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence


# In[27]:


num_words = 10000
max_len = 500
batch_size = 32


# In[28]:


(x_train, y_train), (x_test,y_test) = imdb.load_data(num_words = num_words)


# In[30]:


print(len(x_train))
print(len(y_test))


# In[31]:


pad_x_train = sequence.pad_sequences(x_train, maxlen= max_len)
pad_x_test = sequence.pad_sequences(x_test, maxlen= max_len)


# In[33]:


print(pad_x_train.shape)
print(pad_x_test.shape)


# In[35]:


model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# In[38]:


model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()


# # 모델 학습

# In[40]:


history = model.fit(pad_x_train, y_train,
                   epochs=10,
                   batch_size=128,
                   validation_split=0.2)


# # 시각화

# In[41]:


import matplotlib.pyplot as plt


# In[42]:


plt.style.use('seaborn-white')


# In[44]:


acc =  history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


# In[45]:


ephochs = range(1, len(loss) + 1)


# In[49]:


plt.plot(ephochs, loss, 'b-', label='training loss')
plt.plot(ephochs, val_loss, 'r:', label='validation loss')
plt.grid()
plt.legend()

plt.figure()
plt.plot(ephochs, acc, 'b-', label='training accuracy')
plt.plot(ephochs, val_acc, 'r:', label='validation accuracy')
plt.grid()
plt.legend()

plt.show()


# # 검증

# In[50]:


model.evaluate(pad_x_test,y_test)


# 전체 시퀀스 아니라 순서대로 500개의 단어만 입력했기 떄문에 성능이 낮게 나온다
# * SimpleRNN은 긴 시퀀스를 처리하는데 적합하지 않다

# In[ ]:




