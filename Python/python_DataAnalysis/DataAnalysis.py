import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

plt.plot(np.random.randn(50).cumsum())


data = [1,2,3]

def append_element(some_list, element):
    some_list.append(element)
    
    
append_element(data, 4)

print(data)


a = [1,2,3]

for i in a:
    print(i)

dir(a)

a.__len__()

len(a)

#iterable,iterator iter()

#mutable
#immutable: (),str()

a= "Hello"

a[0]
a[1]
a[1] =  "E"

s = "hoho\nAJC"
print(s)

s = "hoho\\AJC"
print(s)

s = r"hoho\AJC"
print(s)

a = '{0:.2f}{1} are worth US${2}'

a.format(4.5560, 'Argentine Pesos',1)


import datetime as dt
## 객체 생성 없이 쓸수있는 메소드 = 클래스 메소드 
d = dt.datetime.now()

d.year
d.month
d.day
d.hour
d.minute
d.second

d.date()

dt.datetime(2022, 3, 24)
dt.date(2022, 3, 24)


d1 = d.strftime('%Y %m %d %H:%M')
d.strptime(d1, '%Y %m %d %H:%M')

dt1 = dt.datetime.now()
dt2 = dt.datetime(2022, 3, 26)

dt2 - dt1



for i in range(4):
    for j in range(4):
        if j > i:
            break
        print(i,j)


obj = enumerate(['a','b','c'])

for i in obj:
    print(i)

for i,j in enumerate(['a','b','c']):
    print(i,j)


sum = 0
for i in range(10000):
    if i % 3 == 0 | i % 5 == 0:
        sum += i
        
print(sum)


(a,b,c)=(4,5,6)

a,b=1,2
b,a=a,b
print(a,b)


a,b = 1,1
a,b=a+b,a # 2,1
a,b=a+b,a # 3,2
a,b=a+b,a # 5,3
a,b=a+b,a # 8,5

a,b=1,1
for i in range(8):
    a,b = a+b,a
    print(a,b)


# 피보나치 함수

def fb(num):
    a,b=0,1
    while a < num:
        print(a,end=",")
        a,b=b,a+b
        
print(fb(100))


seq = [(1,2,3),(4,5,6),(7,8,9)]

for a,b,c in seq:
    print('a={0},b={1},c={2}'.format(a,b,c))
    
    
a,b,*c=1,2,3,4,5

print(a,b,c)


tu = (1,2,3,3,3,2)
tu.count(2)
tu.index(1)

### 0325

# 문제
df = pd.read_csv("csv_exam.csv")
df
# 과학성적이 60인 학생에 대하여 수학성적의 반별 평균은?
df.rename({"class":"classes"},axis = 1,inplace =True)
df.query('science >= 60').groupby('classes').mean()['math']

#date
import datetime
datetime.datetime.now()

a,b=1,2
b,a=a,b
print(a,b)


# 리스트

li = ['a','b','c','d']
li.append('e')
li.insert(3, "A")
li.pop(3)

li.append("a")
li.remove("a")
li.sort()

'a' in li
'f' not in li

li1 = ["A","B","C"]
li + li1
li.append(li1)
li.extend(li1)

li = ['ABCD','a','ab','abc','a']
li.sort(key = len)


li = ['b','a','d','c']
li.sort(reverse=True)

li = ['b','a','d','c']
li.sort()
li[::-1]

# enumerate

for i,j in enumerate(['a','b','c','d']):
    print("intdex",i,"value",j)

d = {"a":"A","b":"B"}
d.keys()
d.values()
d.items()
d["b"]
# "c":"C"

d['c'] = "C"

li = ['a','b','c','d']

d = {}

for i,j in enumerate(li,start=1):
    d[j] = i
print(d)

li = zip([1,2,3],["a","b","c"],["A","B","C"])

print(list(li))

for i,j,k in zip([1,2,3],["a","b","c"],["A","B","C"]):
    print(i,j,k)
    
# dictionary    
hash('string')

hash((1,2,(2,3)))

# 집합


# 리스트 표기

a = [1,2,3,4,5,6]
b = []
for i in a:
    if i % 2 == 0:
      b.append(i) 
print(b)

[ i for i in a if i % 2 == 0]

strings = ['a','as','bat','car','dove','python']
newli = []
for i in strings:
    if len(i) > 2:
        newli.append(i.upper())

[ i.upper() for i in strings if len(i) > 2]

[newli.append(i.upper()) for i in strings if len(i) > 2]

new_dict = {}
for i in strings:
    if len(i)>2:
        new_dict[i[0]] = i.upper()

dic={new_dict[i[0]]: i.upper() for i in strings  if len(i)>2}


li = [['a','b','c'],["A","B","C"]]

for i in range(len(li)):
    for j in range(len(li[i])):
        print(li[i][j])
        
for i in [0,1]:
    for j in [0,1,2]:
        print(li[i][j])
        
li_new = []
for i in range(len(li)):
    for j in range(len(li[i])):
        li_new.append(li[i][j])
        
[li[i][j] for i in range(len(li)) for j in range(len(li[i]))]
## 함수
# map을 사용하여 모두 숫자로 바꾸어 출력
li = ["1","2","12","123"]

for i in map(int, li):
    print(i)
    
# 익명,람다 함수
def apply_to_list(some_list, f):
    return [f(x) for x in some_list]

li = [1,2,3,4]

def twice(x):
    return x*2

apply_to_list(li, twice)

# 제너레이터

sum( x ** 2 for x in range(100))

dict((i, i ** 2) for i in range(5))

# 파일 읽그 쓰기 

f = open("test.txt", mode = "wb")
f.write()
f.close()

with open("test.txt", mode = "rb") as f:
    f.read()
    f.readline()
    f.readlines()
    
# numpy



import numpy as np
 
# 표준 정규분표
np.random.randn(2,3)

# u = 80, sd = 3
# 정규 분표

np.random.randn(2,3)*3+80

np.random.seed(1234)
data = np.random.randn(4,4)

data[(1,3),]
data[1::2,]

mask = [False,True,False,True]
data[mask]

mask = np.array([0,1,0,1],dtype = 'bool')
data[mask]

data[data>0]

data[data<0] = 0

a = np.array([1,2,10,11])
b = np.array([2,1,10,13])
np.max(a)
np.max(b)
np.maximum(a) # error
np.maximum(a,b)

a = np.array([-2.1,-1.5,0.3,1.5,2.7])
np.modf(a)

np.mod(a,b)

a/b
a//b

a = np.array([1,2,3])
b = np.array([3,4,5])

points = np.arange(-5,5-0.01)
xs, ys = np.meshgrid(a,b)
z = np.sqrt(xs ** 2 + ys ** 2)
z
import matplotlib.pyplot as plt
plt.scatter(xs,ys)
z = np.sqrt(xs**2+ys**2)
plt.imshow(z); plt.colorbar()


points = np.arange(-5,5,0.01)
xs, ys = np.meshgrid(points,points)
z = np.sqrt(xs **2 + ys**2)
plt.imshow(z);
plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")

## 0328

a = np.array([1,2,3])
b = np.array([3,4,5])
x ,y = np.meshgrid(a,b)

plt.scatter(x, y)
plt.grid()

# np.where() <- ifelse (조건, 참일떄, 거짓일 때)

xarr = np.array([1.1, 1.2, 1.3, 1.4, 1.5])

np.where(xarr > 1.3, 10, 0)

np.random.seed(111)
arr = np.random.randn(5,5)

np.where(arr > 0, True, False).sum()
np.where(arr > 0, 1, 0).sum()

np.random.seed(11)
arr = np.random.randn(5,5)*10
arr.sort(axis = 0)
arr

arr1 = np.array([[2,4],[1,3]])
np.linalg.inv(arr1)
np.linalg.det(arr1)
np.trace(arr1)


arr = np.array([[2,4],[1,8]])
arr1 = np.array([6,10])

np.linalg.solve(arr,arr1)

np.diag(np.array([1,2,3]))


# np.random

# 1) 표준정규분포에서
np.random.randn( 4, 4)
np.random.normal(size = ( 4, 4))
np.random.normal(4)
np.random.normal(4,2)

# 2) [0,1] 사이에서 균등분
np.random.rand(2)
np.random.rand(3)
np.random.rand(2,2)

# 3)
np.random.randint(2)
np.random.randint(2,4)
np.random.choice(5)

np.random.sample(10)

import random

position = 0
walk = [position]
steps = 1000
random.seed(100)
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    print("step:", step)
    position += step
    print("position: ", position)
    walk.append(position)
    
plt.plot(walk[:100])

walk.sort()
max(walk)
min(walk)


# pandas

df = pd.read_csv("csv_exam.csv")
df.rename({"class":"classes"}, axis = 1, inplace = True)

df['mean'] = (df['math'] + df['english'] +df['science']) / 3

# 평균이 70점 이상이면 합격, 그렇지 않으면 불합격

df['result'] = np.where(df['mean'] >= 70 , "합격", "불합격"  )

df.query('math > 50 & english < 80')

import plotly.express as px
df = px.data.gapminder()

df.columns
df['country'].nunique()
# Canada만 모으기
np.where(df['country'] == 'Canada', True, False).sum()
df[df['country']=='Canada'].value_counts().value_counts()
df.query('country=="Canada"')
mask = df['country']=='Canada'
df[mask]

# Canada의 1980 이후 자료는?
df.query('country == "Canada" and year >= 1980')


# Asia 국가를 lifeExp 순서로
df[(df['continent'] == 'Asia') & (df['year'] == 1982)].sort_values('lifeExp',ascending=False
).iloc[:,:5]

mask = df['country'].str.contains("Canada")
df[mask]

mask = df['country'].str.contains("Canada|Korea")
df[mask]

df['country'].rename({'Korea, Dem. Rep.':'North Korea'})

df['country'] = np.where(df['country'] == "Korea, Dem. Rep.", "North Korea", df['country'])

df[df['country'] == "North Korea"]



###
Ser = pd.Series([1,3,2,4], index = ["a","c","b","d"])
Ser.reindex(index = ['a','b','c','d'])


Ser = pd.Series(['blue','purple','yellow'], index = [0, 2, 4])

Ser.reindex(range(6))
Ser.reindex(range(6), method = 'ffill')

df.iloc[:, [1,0,2,3,4,5]]


# year contry gdpPercap lifeExp
df1=df.copy()
df1.columns
df1.iloc[:, [2, 0, 5, 3]]
df[['year','country','gdpPercap','lifeExp']]
df[:2]
df[2:4]
df.iloc[:,2:4]

df[-5:]


##
df1.iloc[1,1] = np.nan
##df1.iloc[1,1] = NULL
##df1.iloc[1,1] == NULL
df1.iloc[1,1] == np.nan

##

arr = np.arange(12).reshape(3,4)

np.sum(arr)
np.sum(arr,axis = 0)
np.sum(arr,axis = 1)

df = pd.DataFrame(arr, columns = ['a','b','c','d'])

df.values

df.max

# 최대값과 최솟값의 차이를 구하는 함수
f = lambda x: x.max() - x.min()

df.apply(f, axis= 0)
df.apply(f, axis= 1)


# 0329

f = lambda x: x * x
df.apply(f)


data = {
    '영화' : ['명량', '극한직업', '신과함께-죄와 벌', '국제시장', '괴물', '도둑들', '7번방의 선물', '암살'],
    '개봉 연도' : [2014, 2019, 2017, 2014, 2006, 2012, 2013, 2015],
    '관객 수' : [1761, 1626, 1441, 1426, 1301, 1298, 1281, 1270], # (단위 : 만 명)
    '평점' : [8.88, 9.20, 8.73, 9.16, 8.62, 7.64, 8.83, 9.10]
}
df = pd.DataFrame(data)
df['영화']
df[['영화','평점']]
df[['영화','개봉 연도']]
df[df['개봉 연도'] > 2015][['영화','개봉 연도']]
df[['영화','개봉 연도']].query('`개봉 연도` > 2015')
df['추천 점수'] = df['관객 수'] * df['평점'] /100
df.sort_values('개봉 연도',ascending=False)


df = pd.DataFrame(np.random.randn(3,3))
df[1].corr(df[2])
df.corr()


data = pd.DataFrame({'a':[1,3,4,3,4],'b':[2,3,1,2,3],'c':[1,5,2,4,4]})

data['a'].value_counts()
data['b'].value_counts()
data['c'].value_counts()

result = data.apply(pd.value_counts).fillna(0)


# 데이터 로딩과 저장, 파일 형식

tables = pd.read_html('C://python//FDIC_Failed_Bank_List.html')

len(tables)

failures = tables[0]

failures.head()



import requests

url = 'https://api.github.com/repos/pandas-dev/pandas/issues'

resp = requests.get(url)

data = resp.json()

data[0]['title']

issues = pd.DataFrame(data, columns=['number','title','labels','state'])
issues = pd.DataFrame(data)

issues[['number','title']]

# 데이터 정제
df = pd.DataFrame(np.random.randn(7,3))
df.iloc[:4, 1] = np.nan
df.iloc[:2, 2] = np.nan

data = pd.Series([1 , np.nan , 3, np.nan , 5])

df.dropna()
df.dropna(thresh=1)
df.dropna(thresh=2)
df.dropna(thresh=4)

arr = np.array(range(10,50))
np.identity(3)


np.full([5,3],1)
z = np.random.random(10)
z[z.argmax()] = 0
print(z)


# index를 a,b,c로 바꿔주세요

df = pd.DataFrame(np.arange(9).reshape(3,3))
df.index = ['a','b','c']
df

df.index = df.index.str.upper()
df

df.index = ['a','b','c']
transform = lambda x: x[:4].upper()
df.index = df.index.map(transform)

df.rename(index=str.lower)

# random

a = [0,1,4,6,7,9]
bins = [0,4,8,10]
a1 = pd.cut(a,bins,right=False)
pd.value_counts(a1)

np.random.seed(100)
a = np.random.randint(100, size=(30))

bins = [0, 20, 40, 60, 80, 100]

a1 = pd.cut(a, bins)
a1.codes

a2 = pd.qcut(a, 4)

pd.value_counts(a1)
pd.value_counts(a2)

a = pd.read_csv('C:/python/csv_exam.csv')

sampler = np.random.permutation(20)

a.take(sampler)

a.sample(n = 10)


# 더미 변수
m = {'key':['b','b','a','c','a','b'],
     'datal':range(6)}
df = pd.DataFrame(m)


pd.get_dummies(df['key']) # 더미변수
pd.get_dummies(df['key']).values # 배열로 만들기


np.random.seed(12345)

values = np.random.rand(10)

values

bins = [0, 0.2, 0.4, 0.6, 0.8, 1]

pd.get_dummies(pd.cut(values, bins))


s = input("입력하세요: ").split(',')

val = 'a,b, guido'

pieces = [x.strip() for x in val.split(',')]

first, second, third = pieces

"::".join(pieces)

val.index(',')

val.find(':')

first.find("a")
first.index("a")
first.endswith("a")
first.startswith("a")

# 정규표현식

# 0330

df = pd.DataFrame(np.full((3,2), 1.0))
df.cumsum(axis=1)


df = pd.DataFrame(['a','b','c','a','b','c'],columns= ["A"])
df.duplicated()
df.drop_duplicates()



df = pd.DataFrame(np.full((3,3), 100.0),columns=['a','b','c'])
df.iloc[0, 2] = np.nan
df.iloc[1, :2] = np.nan
df.iloc[2, 0] = np.nan
df

df.fillna(0, inplace = True)


arr = np.random.randint(100,size=35)

pd.qcut(arr, 5).value_counts()
pd.qcut(arr, 5).describe()


pd.qcut(arr, 4) # 4분위 quartile
pd.qcut(arr, 10) # 10분위 quantile

pd.read_csv('C:/python/csv_exam.csv',nrow = 100)


a,b,c = map(int,input('입력: ').split(','))
a+b+c

data = pd.Series([1, np.nan , 3 , np.nan, 5])
data.dropna()

data.fillna(method = 'ffill')
data.fillna(method = 'bfill')
data.fillna(data.mean())



# 계층적 색인

frame = pd.DataFrame(np.arange(12).reshape((4,3)),
                     index=[['a','a','b','b'],[1,2,1,2]],
                     columns=[['Ohio','Ohio','Colorado'],
                              ['Green','Red','Green']])


frame.index
frame.index.names = ['key1', 'key2']
frame.columns
frame.columns.names = ['state', 'color']

frame['Ohio']

frame.swaplevel('key1', 'key2')

frame.sort_index(level=1)
frame.sum(level='key2')
frame.sum(level= 1)
frame.sum(level= 0)
frame.sum(level= 0, axis = 1)
frame.sum(level= 1, axis = 1)

# 계층의 순서를 바꾸고 정렬하기
dict1 = {"a": range(7),"b":range(7,0,-1),
         "c": ['one','one','one','two','two','two','two'],
         "d": [0,1,2,0,1,2,3]}

df = pd.DataFrame(dict1)
df = df.set_index("b")
df
df.index
df = df.reset_index()

df = df.set_index(['a','b'])

df.reset_index(level = 1)
df = df.reset_index()


frame.reset_index(level=1)
frame.reset_index(level=0)
frame.reset_index()

# 데이터 합치기


df1 = pd.DataFrame({'key': ['b','b','a','c','a','a','b'],
                    'data1':range(7)})

df2 = pd.DataFrame({'key': ['a','b','d'],
                    'data2':range(3)})

df1
df2

pd.merge(df1,df2)
pd.merge(df1,df2, how = 'inner', on = 'key')
df1.merge(df2, how = 'inner', on = 'key')
pd.merge(df1,df2, how = 'outer', on = 'key')
pd.merge(df1,df2, how = 'left', on = 'key')
pd.merge(df1,df2, how = 'right', on = 'key')
pd.merge(df1,df2, how = 'cross')

### 연습 문제

df = pd.DataFrame(np.arange(1,10).reshape(3,3),index=[['A','A','B'],['a','b','a']],
                  columns=[['X','X','Z'],['x','y','z']])
df.index.names = ['ㄱ','ㄴ']
df.columns.names = ['가','나']

df.swaplevel('ㄱ','ㄴ')
df.swaplevel('가','나',axis = 1)

df.sum(level=0)
df.sum(level=1)
df.sum(level=1,axis=1)

df = pd.DataFrame(np.arange(1,17).reshape(4,4), columns= ['a','b','c','d'])

df = df.set_index('a')
df = df.reset_index()

df = df.set_index(['a','b'])
df = df.reset_index('b')

a = ['p','y','t','h','o','n']

ap = [x.strip() for x in a.split(',')]
'::'.join(a)
''.join(a)



##

df1 = pd.DataFrame(np.arange(6).reshape(3,2), index = ['a','b','c'],
                   columns=['one','two'])

df2 = pd.DataFrame(np.arange(4).reshape(2,2), index = ['a','c'],
                   columns=['three','four'])

df1
df2


pd.concat([df1,df2], axis=1,keys=['level1','level2'])
pd.concat({'level1': df1, 'level2': df2}, axis=1)

pd.concat([df1,df2], axis=1, keys=['level1','level2'],
          names=['upper','lower'])

pd.concat([df1,df2], axis=1, ignore_index=True)


df = pd.DataFrame(np.arange(1,17).reshape(4,4),
                  index=['A','A','B','B'],
                  columns= [['a','a','b','b'],['c','c','d','d']])

df.stack()
df = df.stack(level=0)
df.unstack(level=0)

# 긴형식에서 넓은 형식으로 피벗하기

data = pd.read_csv('macrodata.csv')

data.head()

periods = pd.PeriodIndex(year=data.year, quarter=data.quarter,
                         name='date')

columns = pd.Index(['realgdp','infl','unemp'], name='item')

data = data.reindex(columns=columns)

data.index = periods.to_timestamp('D','end')

ldata = data.stack().reset_index().rename(columns={0: 'value'})

ldata[:10]

pivoted = ldata.pivot('date','item','value')

pivoted


ldata['value2'] = np.random.randn(len(ldata))

ldata[:10]

ldata.pivot('date','item')
ldata.pivot('date','item')['value']
ldata.pivot('date','item')['value2']

ldata.set_index(['date','item']).unstack('item')


# 넓은 형식에서 긴 형식으로 피벗 하기 melt

df = pd.DataFrame({'key':['foo','bar','baz'],
                   'A':[1,2,3],
                   'B':[4,5,6],
                   'C':[7,8,9]})
df

melted = pd.melt(df, ['key'])
melted

reshaped = melted.pivot('key','variable','value')
reshaped

reshaped.reset_index()

pd.melt(df, id_vars=['key'],value_vars=['A','B'])

pd.melt(df, value_vars=['A','B','C'])
pd.melt(df, value_vars=['key','A','B'])



df = pd.read_csv("csv_exam.csv")
df
df = df.rename({"class":"classes"},axis = 1)
df

df.pivot('id','classes','math')


# 문제

frame = pd.DataFrame(np.arange(9).reshape((3,3)),
                     index=[['X','X','Y'],['x1','x2','y1']],
                     columns=[['C','C','D'],[1,2,1]])

frame.index.names = ['A', 'B']

frame.reset_index(inplace=True)

