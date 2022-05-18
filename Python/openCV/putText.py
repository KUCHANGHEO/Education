#!/usr/bin/env python
# coding: utf-8

# In[22]:


import numpy as np
import cv2


# In[23]:


blue, green, red = (255,0,0), (0,255,0), (0,0,255)
image = np.zeros((400,600,3), np.uint8) 
image[:] = (255,255,255)


# In[24]:


pt1, pt2 = (50,230), (50,310)


# In[25]:


cv2.putText(image, "SIMPLEX", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, blue)
cv2.putText(image, "DUPLEX", (150,130), cv2.FONT_HERSHEY_DUPLEX, 2, blue)
cv2.putText(image, "TRIPLEX", pt1, cv2.FONT_HERSHEY_TRIPLEX, 3, green)
fontFace = cv2.FONT_HERSHEY_PLAIN | cv2.FONT_ITALIC # 글자체 상수
cv2.putText(image, 'ITALIC', pt2, fontFace, 4, blue)


# In[26]:


cv2.imshow('Put Text', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




