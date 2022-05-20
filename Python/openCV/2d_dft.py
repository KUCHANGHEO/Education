#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np, cv2, time
from Common.dft2d import dft, idft, calc_spectrum, fftshift


# In[17]:


def dft2(image):
    tmp = [dft(row) for row in image]
    dst = [dft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)                   # 전치 환원 후 반환


# In[18]:


def idft2(image):
    tmp = [idft(row) for row in image]
    dst = [idft(row) for row in np.transpose(tmp)]
    return np.transpose(dst)                   # 전치 환원 후 반환


# In[19]:


def ck_time(mode = 0):
    global stime
    if (mode ==0 ):
       stime = time.perf_counter()
    elif (mode==1):
       etime = time.perf_counter()
       print("수행시간 = %.5f sec" % (etime - stime))   #초 단위 경과 시간


# In[20]:


image = cv2.imread('images/dft_64.jpg', cv2.IMREAD_GRAYSCALE)
if image is None: raise Exception("영상파일 읽기 에러")


# In[21]:


ck_time(0)
dft = dft2(image)                     # 2차원 DFT 수행
spectrum1 = calc_spectrum(dft)
spectrum2 = fftshift(spectrum1)
idft = idft2(dft).real                # 2차원 IDFT 수행
ck_time(1)


# In[22]:


cv2.imshow("image", image)
cv2.imshow("spectrum1", spectrum1)
cv2.imshow("spectrum2", spectrum2)
cv2.imshow("idft_img", cv2.convertScaleAbs(idft))
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




