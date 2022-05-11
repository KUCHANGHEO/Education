# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 09:23:17 2022

@author: user
"""

import pandas as pd

CCTV_RESULT = pd.read_csv("CCTV_RESULT.csv")
CCTV_IN_SEOUL = pd.read_csv("CCTV_IN_SEOUL.csv")
population_in_seoul = pd.read_excel("population_in_seoul.xls")

CCTV_IN_SEOUL.info()

CCTV_IN_SEOUL['기관명']
CCTV_IN_SEOUL = CCTV_IN_SEOUL.rename({"기관명":"구별"},axis = 1)


population_in_seoul.info()

# 칼럼 솎아내기
pops = population_in_seoul[['자치구','인구','인구.3','인구.6','65세이상고령자']]

pops.info()

# 인덱스 솎아내기

pops = pops.drop([0,1,2])


# index 번호를 0번 부터 다시 나오더록

pops = pops.reset_index()
pops = pops.iloc[:,1:]
pops

pops.columns = ["구별","인구수","한국인","외국인","고령자"]


# 인구수가 3번째 많은 구는?

pops.sort_values(by = '인구수', ascending=False)

pops = pops.dropna() # NaN있는행 삭제

# 구별 인구수 대비 외국인비율, 고령자 비율 2가지 새룬 칼럼

pops['외국인비율'] = pops['외국인'] / pops['인구수'] * 100
pops['고령자비율'] = pops['고령자']/pops['인구수']*100

# 구별 인구수를 막대그래프로 나타내 봅시다

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='Malgun Gothic')


pops.plot(kind = 'bar', x = '구별', y = '인구수')
plt.xticks(rotation=55)
sns.barplot(data = pops , x = '구별' , y = '인구수')
plt.xticks(rotation=55)

# 한국인과 고령자의 상관 계수는?
pops = pops.astype({'인구수' : 'int', '한국인' : 'int', '외국인' : 'int', '고령자' : 'int', '외국인비율' : 'float', '고령자비율' : 'float'})
pops['한국인'].corr(pops['고령자'])
pops[["한국인","고령자"]].astype("int").corr()

lambda pops: pops.astype('float')
(lambda x: pops.iloc[:, 1:].astype(x))('float')
pops = pops.iloc[:,:1].join(pops.iloc[:, 1:].apply(lambda x : x.astype(float)))
pops.info()

pops.corr()

#####


def logistic(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)

plt.plot(x, logistic(x))


#####


CCTV_IN_SEOUL

CCTV_IN_SEOUL.sort_values(by = '소계', ascending=False).iloc[2]
CCTV_IN_SEOUL.sort_values(by = '소계', ascending=False)['구별'].reset_index(drop=True)[2]

CCTV_IN_SEOUL['최근 증가율']=round((CCTV_IN_SEOUL['2016년']/(CCTV_IN_SEOUL['2014년'] + CCTV_IN_SEOUL['2015년'])-1)*100,1)


data = pops.merge(CCTV_IN_SEOUL, how = 'inner', on = '구별')


data = data.set_index('구별')

plt.xticks(rotation=55)
data['인구수'].plot(kind = 'bar')

plt.xticks(rotation=55)
sns.barplot(x = data.index, y='인구수', data=data)


data.sort_values('소계'/'인구수')


data= data.reset_index()
data['cctv비율']= data['소계']/data['인구수']

data.sort_values(by='cctv비율', ascending=False).reset_index(drop=True)

data.plot(kind = 'scatter', x = '인구수', y = '소계')

data.plot(kind = 'barh', y = '소계')
