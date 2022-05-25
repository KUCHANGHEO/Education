#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.datasets import imdb
import numpy as np
import matplotlib.pyplot as plt


# In[5]:


(x_train, y_train), (x_test,y_test) = imdb.load_data()

print('train Data count : {}'.format(len(x_train)))
print('test Data count : {}'.format(len(x_test)))
num_classes = len(set(y_train))
print('classes : {}'.format(num_classes))


# In[9]:


x_train


# In[11]:


print(' train Data review :', x_train[0])
print(' train Label review :', y_train[0])


# In[15]:


review_length = [len(review) for review in x_train]

print('max review length : {}'.format(np.max(review_length)))
print('mean review length : {}'.format(np.mean(review_length)))
print('min review length : {}'.format(np.min(review_length)))


# In[16]:


plt.subplot(1,2,1)
plt.boxplot(review_length)
plt.subplot(1,2,2)
plt.hist(review_length, bins=50)
plt.show()


# In[18]:


word_to_index = imdb.get_word_index()
index_to_word = {}
for key, value in word_to_index.items():
    index_to_word[value+3] = key


# In[21]:


print('빈도수 상위 1등 단어: {}'.format(index_to_word[4]))


# In[22]:


print('빈도수 상위 100등 단어: {}'.format(index_to_word[104]))


# In[27]:


for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
    index_to_word[index] = token
    
print(" ".join([index_to_word[index] for index in x_train[0]]))


# In[ ]:




