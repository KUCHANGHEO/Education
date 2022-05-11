#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2


# In[7]:


switch_case = {
    ord("a"): " a키 입력",
    ord("b"): " b키 입력",
    0x41: "A키 입력",
    int('0x42', 16): "B키 입력",
    2424832: "왼쪽 회살표키 입력",
    2490368: "윗쪽 회살표키 입력",
    2555904: "오른쪽 회살표키 입력",
    2621440: "아래쪽 회살표키 입력"
}

image = np.ones((200,300), np.float)
cv2.namedWindow("Keaboard Event")
cv2.imshow("Keyboard Event", image)
while True:
    key = cv2.waitKeyEx(100)
    if key == 27:break
    try:
        result = switch_case[key]
        print(result)
    except KeyError:
        result = -1
cv2.destroyAllWindows()


# In[ ]:




