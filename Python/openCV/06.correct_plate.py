#!/usr/bin/env python
# coding: utf-8

# In[1]:


from header.plate_preprocess import *        # 전처리 및 후보 영역 검출 함수
from header.plate_candidate import *         # 후보 영역 개선 및 후보 영상 생성 함수


# In[2]:


car_no = 0
image, morph = preprocessing(car_no)  # 전처리 - 이진화
candidates = find_candidates(morph)  # 번호판 후보 영역 검색


# In[3]:


fills = [color_candidate_img(image, size) for size, _, _ in candidates]
new_candis = [find_candidates(fill) for fill in fills]
new_candis = [cand[0] for cand in new_candis if cand]
candidate_imgs = [rotate_plate(image, cand) for cand in new_candis]


# In[6]:


for i, img in enumerate(candidate_imgs):
    cv2.imshow("candidate_img - " + str(i), img)
    cv2.polylines(image, [np.int32(cv2.boxPoints(new_candis[i]))], True, (0, 255, 0), 2)
cv2.imshow("image", image)
cv2.waitKey()  # 키 이벤트 대기
cv2.destroyAllWindows()


# In[ ]:



