#!/usr/bin/env python
# coding: utf-8

# In[2]:


from header.plate_preprocess import *               # 전처리 및 후보 영역 검출 함수


# In[3]:


car_no = int(input("자동차 영상 번호 (0~15): "))
image, morph = preprocessing(car_no)                               # 전처리 - 이진화
if image is None: Exception("영상 읽기 에러")


# In[4]:


candidates = find_candidates(morph)                        # 번호판 후보 영역 검색
for candidate in candidates:                                      # 후보 영역 표시
    pts = np.int32(cv2.boxPoints(candidate))
    cv2.polylines(image, [pts], True, (0, 225,255), 2)
    print(candidate)


# In[6]:


if not candidates:
    print("번호판 후보 영역 미검출")
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




