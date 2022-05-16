#!/usr/bin/env python
# coding: utf-8

# In[30]:


import tensorflow as tf


# In[5]:


mnist =  tf.keras.datasets.mnist


# In[31]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[32]:


X_train.shape


# In[19]:


X_train, X_tset = X_train / 255.0, X_test / 255.0


# In[20]:


X_train


# In[24]:


del model


# In[25]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# In[26]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[27]:


model.fit(X_train,y_train, epochs=10)


# In[28]:


model.save('model3.h5')


# In[ ]:




