#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


# In[4]:


total_words = 10000
batch_size = 128
max_review_len = 80
embedding_len = 100


# In[5]:


(x_train, y_train), (x_test,y_test) = imdb.load_data(num_words = total_words)


# In[6]:


print(x_train.shape)
print(x_test.shape)


# In[7]:


pad_x_train = sequence.pad_sequences(x_train, maxlen= max_review_len)
pad_x_test = sequence.pad_sequences(x_test, maxlen= max_review_len)


# In[8]:


print(pad_x_train.shape)
print(pad_x_test.shape)


# In[16]:


train_data = tf.data.Dataset.from_tensor_slices((pad_x_train,y_train))
train_data = train_data.shuffle(10000).batch(batch_size, drop_remainder=True)
test_data = tf.data.Dataset.from_tensor_slices((pad_x_test,y_test))
test_data = test_data.batch(batch_size, drop_remainder=True)
print('x_train_shape:', pad_x_train.shape, tf.reduce_max(y_train), tf.reduce_min(y_train))
print('x_test_shape:', pad_x_test.shape)


# In[35]:


class LSTM_Build(tf.keras.Model):
    def __init__(self, units):
        super(LSTM_Build, self).__init__()
        
        self.state0 = [tf.zeros([batch_size, units]), tf.zeros([batch_size,units])]
        self.state1 = [tf.zeros([batch_size, units]), tf.zeros([batch_size,units])]
        
        self.embedding = tf.keras.layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        self.RNNCell0 = tf.keras.layers.LSTMCell(units, dropout=0.5)
        self.RNNCell1 = tf.keras.layers.LSTMCell(units, dropout=0.5)
        self.outlayer = tf.keras.layers.Dense(1)
        
    def call(self, inputs, training=None):
        x = inputs
        x = self.embedding(x)
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):
            out0, state0 = self.RNNCell0(word, state0, training)
            out1, state1 = self.RNNCell1(out0, state1, training)
        x = self.outlayer(out1)
        prob = tf.sigmoid(x)
        return prob


# In[36]:


import time
units = 64
epochs = 4
t0 = time.time()

model = LSTM_Build(units)

model.compile(optimizer=Adam(0.001),
             loss=tf.losses.BinaryCrossentropy(),
             metrics=['accuracy'],
             experimental_run_tf_function=False)


# In[37]:


model.fit(train_data, epochs=epochs, validation_data=test_data, validation_freq=2)


# In[38]:


print("훈련 데이터셋 평가...")
(loss, accuracy) = model.evaluate(train_data, verbose=0)
print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
print("테스트 데이터셋 평가...")
(loss, accuracy) = model.evaluate(test_data, verbose=0)
print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
t1 = time.time()
print('시간:', t1-t0)


# In[52]:


class LSTM_Build(tf.keras.Model):
    def __init__(self, units):
        super(LSTM_Build, self).__init__()
        
        self.state0 = [tf.zeros([batch_size, units]), tf.zeros([batch_size,units])]
        self.state1 = [tf.zeros([batch_size, units]), tf.zeros([batch_size,units])]
        
        self.embedding = tf.keras.layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        self.rnn = tf.keras.Sequential([
            tf.keras.layers.LSTM(units, dropout=0.5, return_sequences=True, unroll=True),
            tf.keras.layers.LSTM(units, dropout=0.5, unroll=True)])
        self.outlayer = tf.keras.layers.Dense(1)
        
    def call(self, inputs, training=None):
        x = inputs
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.outlayer(x)
        prob = tf.sigmoid(x)
        return prob


# In[53]:


import time
units = 64
epochs = 4
t0 = time.time()

model = LSTM_Build(units)

model.compile(optimizer=Adam(0.001),
             loss=tf.losses.BinaryCrossentropy(),
             metrics=['accuracy'],
             experimental_run_tf_function=False)


# In[54]:


model.fit(train_data, epochs=epochs, validation_data=test_data, validation_freq=2)


# In[55]:


print("훈련 데이터셋 평가...")
(loss, accuracy) = model.evaluate(train_data, verbose=0)
print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
print("테스트 데이터셋 평가...")
(loss, accuracy) = model.evaluate(test_data, verbose=0)
print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
t1 = time.time()
print('시간:', t1-t0)


# In[56]:


model.summary()


# In[57]:


class GRU_Build(tf.keras.Model):
    def __init__(self, units):
        super(GRU_Build, self).__init__()
        
        self.state0 = [tf.zeros([batch_size, units]), tf.zeros([batch_size,units])]
        self.state1 = [tf.zeros([batch_size, units]), tf.zeros([batch_size,units])]
        
        self.embedding = tf.keras.layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        self.RNNCell0 = tf.keras.layers.GRUCell(units, dropout=0.5)
        self.RNNCell1 = tf.keras.layers.GRUCell(units, dropout=0.5)
        self.outlayer = tf.keras.layers.Dense(1)
        
    def call(self, inputs, training=None):
        x = inputs
        x = self.embedding(x)
        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):
            out0, state0 = self.RNNCell0(word, state0, training)
            out1, state1 = self.RNNCell1(out0, state1, training)
        x = self.outlayer(out1)
        prob = tf.sigmoid(x)
        return prob


# In[58]:


import time
units = 64
epochs = 4
t0 = time.time()

model = GRU_Build(units)

model.compile(optimizer=Adam(0.001),
             loss=tf.losses.BinaryCrossentropy(),
             metrics=['accuracy'],
             experimental_run_tf_function=False)


# In[59]:


model.fit(train_data, epochs=epochs, validation_data=test_data, validation_freq=2)


# In[62]:


print("GRU 훈련 데이터셋 평가...")
(loss, accuracy) = model.evaluate(train_data, verbose=0)
print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
print("GRU 테스트 데이터셋 평가...")
(loss, accuracy) = model.evaluate(test_data, verbose=0)
print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
t1 = time.time()
print('시간:', t1-t0)


# In[63]:


class GRU_Build(tf.keras.Model):
    def __init__(self, units):
        super(GRU_Build, self).__init__()
        
        self.state0 = [tf.zeros([batch_size, units]), tf.zeros([batch_size,units])]
        self.state1 = [tf.zeros([batch_size, units]), tf.zeros([batch_size,units])]
        
        self.embedding = tf.keras.layers.Embedding(total_words, embedding_len, input_length=max_review_len)
        self.rnn = tf.keras.Sequential([
            tf.keras.layers.GRU(units, dropout=0.5, return_sequences=True, unroll=True),
            tf.keras.layers.GRU(units, dropout=0.5, unroll=True)])
        self.outlayer = tf.keras.layers.Dense(1)
        
    def call(self, inputs, training=None):
        x = inputs
        x = self.embedding(x)
        x = self.rnn(x)
        x = self.outlayer(x)
        prob = tf.sigmoid(x)
        return prob


# In[64]:


import time
units = 64
epochs = 4
t0 = time.time()

model = GRU_Build(units)

model.compile(optimizer=Adam(0.001),
             loss=tf.losses.BinaryCrossentropy(),
             metrics=['accuracy'],
             experimental_run_tf_function=False)


# In[65]:


model.fit(train_data, epochs=epochs, validation_data=test_data, validation_freq=2)


# In[66]:


print("GRU 훈련 데이터셋 평가...")
(loss, accuracy) = model.evaluate(train_data, verbose=0)
print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
print("GRU 테스트 데이터셋 평가...")
(loss, accuracy) = model.evaluate(test_data, verbose=0)
print("loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
t1 = time.time()
print('시간:', t1-t0)


# In[ ]:




