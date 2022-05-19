#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[2]:


font = cv2.FONT_HERSHEY_COMPLEX
fontScale = 2
fontColor = (255, 0, 255)
lineType = 2;


# In[3]:


face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# In[9]:


try:
cap = cv2.VideoCapture(0)

except:
    print('Camera Loading Faile !!!')
    return 

while True:
    ret, frame = cap.read()
    gray = cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Detect Faced", (x-5, y-5), font, 1, (255, 0, 255), 2)
        face_gray = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_color, (ex,ey), (ex+ew,ey+eh), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    key =cv2.waitKey(20) # space bar
    if key == ord('q')
        break

cap.release()
cv2.destroyAllWindows()


# In[ ]:




