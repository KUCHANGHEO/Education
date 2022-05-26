#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
from tensorflow.keras import layers


# In[2]:


df = pd.read_csv('./data/covertype.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


x = df[df.columns[:54]]


# In[6]:


x.info()


# In[7]:


y = df['class']


# In[10]:


y.value_counts()


# In[37]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state = 90)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation = 'relu',
                         input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense (8, activation = 'softmax')
])
model.compile(optimizer = tf.keras.optimizers.Adam(0.001),
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])
history1 = model.fit(x_train, y_train,
                    epochs = 26, batch_size = 60,
                    validation_data = (x_test, y_test))


# In[38]:


import matplotlib.pylab as plt


# In[42]:


acc = history1.history['accuracy']
loss= history1.history['loss']
val_acc = history1.history['val_accuracy']
val_loss = history1.history['val_loss']


# In[43]:


epochs = range(len(acc))


# In[44]:


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


# # 표준화

# In[56]:


from sklearn import preprocessing
df = pd.read_csv('./data/covertype.csv')
x = df[df.columns[:55]]
y = df['class']
x_train, x_test, y_train, y_test = train_test_split(x, y , train_size = 0.7, random_state =  90)
train_norm = x_train[x_train.columns[0:10]]
test_norm = x_test[x_test.columns[0:10]]
std_scale = preprocessing.StandardScaler().fit(train_norm)
x_train_norm = std_scale.transform(train_norm)
training_norm_col = pd.DataFrame(x_train_norm, index=train_norm.index, columns=train_norm.columns)
x_train.update(training_norm_col)
print (x_train.head())
x_test_norm = std_scale.transform(test_norm)
testing_norm_col = pd.DataFrame(x_test_norm, index=test_norm.index, columns=test_norm.columns)
x_test.update(testing_norm_col)
print (x_test.head())


# In[52]:


history2 = model.fit(x_train, y_train,
                    epochs = 26, batch_size = 60,
                    validation_data = (x_test, y_test))


# In[57]:


acc = history2.history['accuracy']
loss= history2.history['loss']
val_acc = history2.history['val_accuracy']
val_loss = history2.history['val_loss']


# In[58]:


epochs = range(len(acc))


# In[59]:


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




