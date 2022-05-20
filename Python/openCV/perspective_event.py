#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, cv2
from Common.utils import contain_pts  # 좌표로 범위 확인 함수


# In[2]:


def draw_rect(img):
    rois = [(p-small, small * 2) for p in pts1]
    for (x,y), (w,h) in np.int32(rois):
        roi = img[y:y + h, x:x + w]                 # 좌표 사각형 범위 가져오기
        val = np.full(roi.shape, 80, np.uint8)  # 컬러(3차원) 행렬 생성		cv2.add(roi, val, roi)                      			# 관심영역 밝기 증가
        cv2.add(roi, val, roi)
        cv2.rectangle(img, (x,y, w, h), (0, 255, 0), 1)
    cv2.polylines(img, [pts1.astype(int)], True, (0, 255, 0), 1)     # pts는 numpy 배열
    cv2.imshow("select rect", img)


# In[3]:


def warp(img):
    perspect_mat = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, perspect_mat, (350, 400), cv2.INTER_CUBIC)
    cv2.imshow("perspective transform", dst)


# In[4]:


def onMouse(event, x, y, flags, param):
    global check
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, p in enumerate(pts1):
            p1, p2 = p - small, p + small           # p점에서 우상단, 좌하단 좌표생성
            if contain_pts((x,y), p1, p2): check = i
    if event == cv2.EVENT_LBUTTONUP: check = -1                                  # 좌표 번호 초기화
    if check >= 0 :                                 # 좌표 사각형 선택 시
        pts1[check] = (x, y)
        draw_rect(np.copy(image))
        warp(np.copy(image))


# In[5]:


image = cv2.imread('images/perspective2.jpg')
if image is None: raise Exception("영상 파일을 읽기 에러")


# In[6]:


small = np.array((12, 12))                                    # 좌표 사각형 크기
check = -1                                          # 선택 좌표 사각형 번호 초기화
pts1 = np.float32([(100, 100), (300, 100), (300, 300), (100, 300)])
pts2 = np.float32([(0, 0), (400, 0), (400, 350), (0, 350)])  # 목적 영상 4개 좌표                         # 목적 영상 4개 좌표


# In[7]:


draw_rect(np.copy(image))
cv2.setMouseCallback("select rect", onMouse, 0)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




