#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2


# In[3]:


cv2.__version__


# In[4]:


image = np.zeros((200,400), np.uint8)


# In[5]:


image[:] = 200


# In[6]:


title1, title2 = 'First Window', 'Second Window'


# In[7]:


cv2.namedWindow(title1, cv2.WINDOW_AUTOSIZE)
cv2.namedWindow(title2)
cv2.moveWindow(title1, 150,150)
cv2.moveWindow(title2, 400, 50)
cv2.imshow(title1, image)
cv2.imshow(title2, image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




