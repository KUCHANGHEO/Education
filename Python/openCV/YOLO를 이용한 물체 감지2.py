#!/usr/bin/env python
# coding: utf-8

# In[4]:


import cv2
import numpy as np


# 알고리즘을 로드합니다. 알고리즘을 실행하기 위해서 세개의 파일이 필요합니다.
# 
# 
# Weight file : 훈련된 model
# 
# Cfg file : 구성파일. 알고리즘에 관한 모든 설정이 있다.
# 
# Name files : 알고리즘이 감지할 수 있는 객체의 이름을 포함한다.

# In[5]:


import os
os.getcwd()


# In[6]:


# Yolo 로드
net = cv2.dnn.readNet("./yolopython/yolov3.weights", "./yolopython/yolov3.cfg")
classes = []
with open("./yolopython/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


# 그 다음 물체 감지를 할 이미지를 로드하고 너비, 높이도 가져옵니다 :

# In[7]:


# 이미지 가져오기
img = cv2.imread("./yolopython/geese.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape


# In[8]:


height, width, channels


# 네트워크에서 이미지를 바로 사용할 수 없기때문에 먼저 이미지를 Blob으로 변환해야 한다.
# 
# Blob은 이미지에서 특징을 잡아내고 크기를 조정하는데 사용된다.
# 
# 
# YOLO가 허용하는 세가지 크기
# 
# 
# - 320 × 320 : 작고 정확도는 떨어지지 만 속도 빠름
# - 609 × 609 : 정확도는 더 높지만 속도 느림
# - 416 × 416 : 중간

# In[9]:


# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)


# outs는 감지 결과이다. 탐지된 개체에 대한 모든 정보와 위치를 제공한다.

# 결과 화면에 표시 / 신뢰도, 신뢰 임계값  계산 (이 부분이 완전 이해가 안된다 ) : 

# In[10]:


# 정보를 화면에 표시
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # 좌표
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


# 신뢰도가 0.5 이상이라면 물체가 정확히 감지되었다고 간주한다. 아니라면 넘어감..
# 
# 
# 임계값은 0에서 1사이의 값을 가지는데
# 
# 1에 가까울수록 탐지 정확도가 높고 , 0에 가까울수록 정확도는 낮아지지만 탐지되는 물체의 수는 많아진다.

# In[11]:


# 노이즈 제거 : 
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


# 같은 물체에 대한 박스가 많은것을 제거
# 
# Non maximum suppresion이라고 한답니다.

# 마지막으로 모든 정보를 추출하여 화면에 표시합니다.
#  
# Box : 감지된 개체를 둘러싼 사각형의 좌표
# 
# Label : 감지된 물체의 이름
# 
# Confidence : 0에서 1까지의 탐지에 대한 신뢰도

# 화면에 표시하기

# In[12]:


font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[13]:


import cv2
import numpy as np

eye_detect = False

face_cascade = cv2.CascadeClassifier('haarcascade_frontface.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# Create a VideoCapture object
cap = cv2.VideoCapture(0)
 
# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")
 
# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 
# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('./yolopython/road.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
 
while(True):
    ret, frame = cap.read()
 
    if eye_detect:
        info = 'Eye detection On'
    else:
        info = 'Eye detection off'
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cv2.putText(frame, info, (5, 10), font, 2, (255, 0, 0), 1)
        
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, 'Detected Face', (x-5, y-5), font, 1, (255, 0, 255), 2)
        if eye_detect:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
        
    # Write the frame into the file 'output.avi'
    out.write(frame)
 
    # Display the resulting frame    
    cv2.imshow('frame',frame)
 
    # Press Q on keyboard to stop recording
    k = cv2.waitKey(20)
        
    if k == ord('i'):
        eye_detect = not eye_detect
    if k == ord('q'):
        break
 
# When everything done, release the video capture and video write objects
cap.release()
out.release()
 
# Closes all the frames
cv2.destroyAllWindows() 


# In[ ]:




