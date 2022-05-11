import pandas as pd
import numpy as np

animals = ['Tigers','Bear','Moose']

# 시리즈
pd.Series(animals)


# 데이터프레임 만들기
# arry
diction = {"a":1,"b":2,"c":3}

pd.Series(diction)

arr = np.array([[1,2,3],[4,5,6]])

pd.DataFrame(arr)

# dictionary
diction = {"a":['1','3'],"b":['1','2'],"c":['2','4']}
pd.DataFrame(diction)

li = [4,5,6,7]
pd.DataFrame(li)
pd.DataFrame(li,index=range(0,4),columns = ["A"])

arr = np.array([[1,2,3],[4,5,6]])
pd.DataFrame(arr)
pd.DataFrame(arr,index = ['A','B'],
             columns = ['a','b','c'])


# 직접만듬

df = pd.DataFrame([[15, '남', '남중'],
                    [17, '여', '여중']],
                  index = ['철수','영희'],
                  columns=['나이','성별','학교'])

li = [[15, '남', '남중'],[17, '여', '여중'],
      [19,'남','남고']]
df = pd.DataFrame(li, index = ['철수','영희','길동'],
                  columns = ["나이","성별","학교"])
print(df)


# df.index = rowname(df)
# df.columns = colnames(df)

## 데이터 프레임의 행과 열 다루기

df.drop('철수')
df.drop(['철수','영희'])

df = pd.DataFrame(li, index = ['철수','영희','길동'],
                  columns = ["나이","성별","학교"])


df = df.drop(['철수','영희']) # 덮어쓰기(방법1)


df.drop(['철수','영희'], inplace=True) # 옵션(방법2)

df.drop(['철수','영희'], axis = 0)
df.drop(['나이','성별'], axis = 1)


# 성별 열을 삭제해 주세요

df.drop('성별',axis = 1)

df = pd.DataFrame([[90,98,85,100],[80,89,95,90],[70,95,100,90]],
             index=['서준','우현','인아'],
             columns=['수학','영어','음악','체육'])

# 표, table, dataframe, DataFrame

# 행 선택

df.loc[['서준','수학']] # location

df.iloc[ 0, 0 ] # integer location

df.loc['서준', :]     # Series
df.loc[['서준'], :]   # DataFrame

df.loc[['서준', '우현']]

df.iloc[[0,1], :]
df.iloc[:2, :]

df['수학']
a = df[['수학']]
type(a)
df[['수학']]
df[['수학','영어']]
df[:]
df[:2]
df[:4]
df.수학[1]
df.head(20)


df = pd.DataFrame([[1,2],[3,4]],index=['a','b'],columns=['A','B'])


b = [0,1]
df.loc["b","B"]
df.loc["b"]["B"]
df.loc["b"][1]
df.loc["B"]["b"] # 이건 안되요
df.loc[['b'],['B']]
df.iloc[1,1]
df.iloc[1][1]
df.iloc[[1, ],[1, ]]
df.B[1]

df['B'][1]
df['B']["b"]


df.index # 새로운 행 이름 리스트
df.columns # 새로운 열 이름 객체 열이름 리스트
df.describe()
df.info()


# DataFrame, ndarray

# 행 이름 바꾸기

df = pd.DataFrame([[90,98,85,100],[80,89,95,90],[70,95,100,90]],
             index=['서준','우현','인아'],
             columns=['수학','영어','음악','체육'])


df.index = ["준","현","아"]

df.columns = ["수","영","음","체"]

df.rename(index = {"서준": "준서", "우현":"현우" })

df.rename(columns = {"수학":"Math","영어":"English"})

import plotly.express as px

df = px.data.gapminder()

df.columns

df.info()

df[:10]
df.country

df1 = df[100:501:100]

df.drop(['gdpPercap','iso_alpha','iso_num'],axis = 1)

df.drop(columns=['gdpPercap','iso_alpha','iso_num'])

df.iloc[:,:5]


df1.index = ["A","B","C","D","E"]

df1.rename(columns = {"continent":"conti","lifeExp":"life"})

df = pd.DataFrame([[1,2],[3,4]],index=['a','b'],columns=['A','B'])

### 0323

b = [0,1]
df.loc["b","B"]
df.loc["b"]["B"]
df.loc["b"][1]
df.loc["B"]["b"] # 이건 안되요
df.loc[['b'],['B']]
df.iloc[1,1]
df.iloc[1][1]
df.iloc[[1, ],[1, ]]
df.B[1]

df['B'][1]
df['B']["b"]

# 행 추가


df = pd.DataFrame([[90,98,85,100],[80,89,95,90],[70,95,100,90]],
             index=['서준','우현','인아'],
             columns=['수학','영어','음악','체육'])
df

df.loc["상기"] = [95, 100, 80, 95]
df

# 열 추가

df["과학"] = [80,90,95,100]
df



# 수학 + 영어 + 과학의 값을 열이름 주요과목 으로 만들기

df["주요과목"] = df["수학"] +df["영어"] +df["과학"]
df

# 원소선택

df.iloc[2,3]
df.iloc[2][3]
df.loc["인아","체육"]
df.loc["인아"]["체육"]

# 인아의 체육, 과학 점수

df.loc["인아",["체육","수학"]]


# 원소선택 = 새로운 값

df.loc["인아"]["체육"] = 90

df.loc["인아",["체육","영어"]] = 80,90
df.loc["인아",["체육","영어"]] = [80,90]



# 파일 읽기

# csv 파일 읽기

df = pd.read_csv("C:/python\csv_exam.csv")
df

address = "https://raw.githubusercontent.com/plotly/datasets/master/auto-mpg.csv"

df = pd.read_csv(address)

df.to_csv("mpg.csv")

# excel 파일 읽기

df1 = pd.read_excel("excel_exam.xlsx")
df1.to_excel("mpg.xlsx")


df.head()
df[:5]
df.tail()
df[-5:]


df.describe()
df.shape
df.info()


# count빈도수
df.count()
df.mpg.count()
df.mpg.value_counts()
df['mpg'].value_counts() #table

df['cylinders']
df['cylinders'].unique
df['cylinders'].value_counts() # 1차원 자료에서 쓰여집니다

# 상관 계수 구하기

df.corr()



# 결측치

import seaborn as sns

df = sns.load_dataset("titanic")
sns.get_dataset_names()

df.head(5)
# r에서는 범주를 factor형 여기선 category형

df[['pclass','class']] # pclass는 숫자형 class는 category형

# 결측치 존재
## NA, NaN
df.isnull()
df.isnull().sum()

df['deck'].isnull().sum()
df['deck'].value_counts()
df['deck'].value_counts(dropna=False)

df.isnull().sum() 
df.info() # index 갯수 891

df1 = df.dropna() # NaN있는행 삭제
df1.info() # index 갯수 182

df2 = df.dropna(axis = 1) # NaN있는 열 삭제
df2.info()

# 결측치 대체
mean_age = round(df['age'].mean(), 1)

df3 = df
df3['age'].fillna(mean_age, inplace = True)
df3.isnull().sum()
df3['age'].isnull()


# 중복 데이터

df = pd.DataFrame({"A":['a','a','b','a','a'],
                  "B":[1,1,1,2,1],
                  "C":[1,1,1,2,1]})

df.duplicated()
df['A'].duplicated
df.drop_duplicates()


# 데이터 합치기

# merge

df1 = pd.DataFrame({'key':['b','b','a','c','a','a','b'],
                    'data1':range(7)})
df1

df2 = pd.DataFrame({'key':['a','b','d'],
                    'data2':['A','B','D']})
df2

pd.merge(df1,df2)
pd.merge(df2,df1)
pd.merge(df1,df2, on = 'key') # 3개는 값이 같
pd.merge(df1,df2, how = 'outer' ,on = 'key')
pd.merge(df1,df2, how = 'inner' ,on = 'key')
pd.merge(df1,df2, how = 'left'  ,on = 'key')
pd.merge(df1,df2, how = 'right' ,on = 'key')
df1.merge(df2)

##

a =pd.read_excel("C:/python/excel_exam.xlsx")
df.to_excel("mpg.xlsx")

a = pd.read_clipboard()
a

b = pd.DataFrame({"class":[1,2,3,4,5],"teacher":["kim","park","jeoung","hong","choi"]})
b

pd.merge(a,b,how="inner", on = 'class',)
pd.merge(a,b,how="outer", on = 'class',)
pd.merge(a,b,how="left", on = 'class',)
pd.merge(a,b,how="right", on = 'class',)



###

a =pd.read_csv("C:/python/csv_exam.csv")
a

a.loc[20, ] = [21, 6, 78, 83, 58]
a.loc[21, ] = [22, 6, 78, 83, 58]
a.loc[22, ] = [23, 6, 78, 83, 58]
a.loc[23, ] = [24, 6, 78, 83, 58]
a['id'] = a['id'].astype('int')

for i in range(5):
    a.iloc[:,i] = a.iloc[:,i].astype('int')


b = pd.DataFrame({"class":[3,4,5,6,7],"teacher":["kim","park","jeoung","hong","choi"]})
b

pd.merge(a,b,how="inner", on = 'class',)
pd.merge(a,b,how="outer", on = 'class',)
pd.merge(a,b,how="left", on = 'class',)
pd.merge(a,b,how="right", on = 'class',)




# concat
E = pd.Series(['e0','e1','e2','e3'],name = 'e')
F = pd.Series(['f0','f1','f2'],name = 'f',index=[3,4,5])
G = pd.Series(['g0','g1','g2','g3'],name = 'g')

pd.concat([E,F])
pd.concat([E,G])
pd.concat([E,G],axis = 1)
pd.concat([E,F],axis = 1)
type(pd.concat([E,G], axis = 0))
type(pd.concat([E,G], axis = 1))

df1 = pd.DataFrame({'a':['a0','a1','a2'],
                    'b':['b0','b1','b2'],
                    'c':['c0','c1','c2']},
                   index = [0,1,2])

df2 = pd.DataFrame({'b':['b0','b1','b2'],
                    'c':['c0','c1','c2'],
                    'd':['d0','d1','d2']},
                   index = [1,2,3])

pd.concat([df1,df2])
pd.concat([df1,df2], ignore_index=True)
pd.concat([df1,df2],axis = 1)
pd.concat([df1,df2],axis = 1, join = 'inner')
pd.concat([df1,df2],axis = 1, join = 'outer')


# group by
df = a.copy()

df.rename({"class":"classes"},axis=1,inplace = True)

# 반별 영어성적의 평균은?
df.groupby(['classes']).mean()
df.groupby(['classes']).mean()['english']
df.groupby(['classes'])['english'].mean()
df[['classes','english']].groupby(['classes']).mean()
df[['classes','english']].groupby(['classes']).mean()
    
df.groupby(['classes']).max()['math']




## 0324
df = pd.DataFrame([[1,2],[6,7]],index=(0,1),columns=("A","B"))
df["C"] = 3,8

df1 = pd.DataFrame([[1,2],[6,7]],index=(0,1),columns=("A","B"))
df1.index = ['a','b']
df1["C"] = 3,8

df1.loc["c"] = [9,10,11]
df.loc[2,:] = [9,10,11]

df["B"]=df["B"].astype('int')


df["합계"]=df["A"]+df["B"]+df["C"]


df.count()
df.C.value_counts()

df.values



## filter

df = pd.read_csv("csv_exam.csv")
df[df>90]

df[df['math'] > 80]

# 영어가 70점 이하인 학생을 골라주세요
df[df['english']<=70]

# 수학은 80점 이상 이거나 영어가 70점 이하인 학생을 골라주세요

df[ (df['math'] > 80) | (df ['english'] <=70) ]

mask = (df['math'] > 80) | (df ['english'] <=70)

df[mask]


# 반이 2반, 3반, 4반 학생중에서 수학은 80점 이상인 학생을 골라주세요

mask = (df['class'] > 1) & (df['class'] < 5) & (df['math'] >= 80)
  
df[mask]

## 
df.rename({"class":"classes"},axis = 1,inplace =True)
df.query('math==50')
df.query('math>50')
df.query('math>50 and english <80')
df.query('classes == 1 and math > 50')
df.query('classes == 1 or classes ==2 and math > 50')
df.query('(classes == 1 or classes ==2) and math > 50')
df.query('(classes == 1 or classes ==2 or classes == 3) and math > 50')
df.query('classes in (2,3,4) and math >= 80 ')

'''
df.query('')
1. ' '
2. and, or , &,| 가능
3. A in B 도 가능
4. 칼럼 이름 바로 사용
'''
# science열을 문자열로 바꾸어 봅시다
df['science'] = df['science'].astype('str')

df.query('science == 90')
df.query('science == 50')
df.query("science == '58'")

# 반은 2,3,4 이고 수학은 50점이상은 몇명?

df.query('classes in (2,3,4) & math >= 50')

# 반별 수학성적의 평균은?
df.groupby(by='classes')['math'].mean()
df.groupby(by='classes').mean()['math']

# 2,3반에 대하여  반별 수학성적의 최대값은?

df.query('classes in [2,3]').groupby('classes')['math'].max()


# 수학성적이 60점인 학생에 대하여 영어성적의 반별 평균은?

df.query('math == 60').groupby('classes').mean()['english']





# 
"""
"""

def add_number(a,b):
    """
    이것은 무엇입니까?
    

    Parameters
    ----------
    a : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return a+b

add_number(1,2)

help(add_number)
