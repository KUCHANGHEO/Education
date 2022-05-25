#!/usr/bin/env python
# coding: utf-8

# In[2]:


# 전이학습에 대하야 알아봅시다
# 비학습된 Pretrained Network VGG16을 이용해 보는것

from tensorflow.keras.applications import VGG16


# In[3]:


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[157]:


# ImageDataGenerator 객체 생성
Datagen = ImageDataGenerator(rescale=1/255)


# In[198]:


# ImageDataGenerator 객체 생성
Train_Datagen = ImageDataGenerator(rescale=1/255)
Test_Datagen = ImageDataGenerator(rescale=1/255)


# In[199]:


data_dir = './data'


# In[200]:


train_dir = './data/train'
test_dir = './data/test'


# In[332]:


# ImageDataGenerator 설정

train_generator = Train_Datagen.flow_from_directory(
    train_dir,               # 학습용 이미지를 가져올 폴더
    classes=['paper','rock','scissors'], # cats 폴더의 이미지 label을 0으로 
                             # dogs 폴더의 이미지는 label을 1으로
    target_size=(150,150),   # 이미지 resize
    batch_size=12,           # 한번에 20개의 이미지만 가져와서 학습
)


test_generator = Test_Datagen.flow_from_directory(
    test_dir,               # 학습용 이미지를 가져올 폴더
    classes=['paper','rock','scissors'], # cats 폴더의 이미지 label을 0으로 
                             # dogs 폴더의 이미지는 label을 1으로
    target_size=(150,150),   # 이미지 resize
    batch_size=12,           # 한번에 10개의 이미지만 가져와서 학습
)


# In[351]:


del model


# In[352]:


model = Sequential()


# In[353]:


# Convolution
model.add(Conv2D(
    filters=32,
    kernel_size=(3,3),
    strides=(1,1),
    activation='relu'))


# In[354]:


model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    strides=(1,1),
    activation='relu'))


# In[355]:


model.add(MaxPooling2D(pool_size=(2,2)))


# In[356]:


model.add(Conv2D(
    filters=64,
    kernel_size=(3,3),
    strides=(1,1),
    activation='relu'))


# In[357]:


# FC layer(DNN) 의: input layer
model.add(Flatten(input_shape=())) # 전체 4차원 에서 2차원으로 바꿔주는것


# In[358]:


# hidden Layer
model.add(Dense(units=256, activation='relu'))


# In[359]:


# output layer
model.add(Dense(units=3, activation='softmax'))


# In[360]:


# model이 어떻게 동작하는지를 지정
model.compile(optimizer=Adam(learning_rate=0.001),
             loss='categorical_crossentropy',
             metrics=['accuracy'])


# In[361]:


# 설정을 모두 마치었으면 모델 학습
history = model.fit(train_generator,
                    steps_per_epoch = 180,
                    epochs = 10,
                    verbose= 1,
                    validation_data = test_generator, validation_steps=31)


# In[362]:


model.summary()


# In[363]:


model.evaluate(test_generator)


# In[364]:


import matplotlib.pyplot as plt


# In[365]:


acc = history.history['accuracy']
loss= history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']


# In[366]:


epochs = range(len(acc))


# In[367]:


plt.plot(epochs, acc, 'bo', label= 'Training accuracy')
plt.plot(epochs, val_acc, 'b', label= 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label= 'Training loss')
plt.plot(epochs, val_loss, 'b', label= 'Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# In[ ]:




