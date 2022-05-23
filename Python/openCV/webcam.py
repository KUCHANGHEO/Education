#!/usr/bin/env python
# coding: utf-8

# # 필요 라이브러리 Import

# In[1]:


import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox


# ## open webcam (webcam 열기)

# In[2]:


webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Could not open webcam")
    exit()


# In[ ]:


while webcam.isOpened():
    status, frame = webcam.read()
    if not status:
        break
    
    # apply object etection(물체 검출)
    bbox, label, conf = cv.detect_common_objects(frame)
    frame2 = draw_bbox(frame, bbox, label, conf, write_conf=True)
    print(bbox, label, conf)
    
    cv2.imshow(" ", frame2)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
webcam.release()
cv2.destroyAllWindows()

