import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import pytz
from sklearn.linear_model import LogisticRegression


data = np.arange(10)
df = pd.DataFrame(data)

plt.plot(df)

df.plot()
# df.plot? df.plot??
df.plot(kind = 'line')
df.plot(kind = 'bar')
df.plot(kind = 'barh')
df.plot(kind = 'box')
df.plot(kind = 'kde')
df.plot(kind = 'area')
# df.plot(kind = 'pie')
# df.plot(kind = 'scatter')

df = pd.DataFrame(np.random.randn(10,4).cumsum(0),
                  columns=['A','B','C','D'],
                  index=np.arange(0,100,10))

df.plot()

df['A']
df['A'].plot(kind = 'bar')
df['A'].plot.bar()
df['A'].plot.barh()
df1 = df[:1]
df1.plot(kind='bar')

df2 = df[:2]
df2.plot(kind='bar')

df3 = df[:3]
df3.plot(kind='bar')

df3.plot(kind='bar',stacked=True)
df3.plot(kind='barh',stacked=True)


import seaborn as sns

sns.get_dataset_names()

tips = sns.load_dataset("tips")

tips.info()

tips['tip_pct'] = tips['tip'] / (tips['total_bill'] - tips['tip'])

# hue = 색상 기준 범례

sns.barplot(data = tips, x = 'day', y = 'tip_pct', hue = 'time')
sns.barplot(data = tips, x = 'day', y = 'tip_pct', hue = 'time')
sns.barplot(x='tip_pct', y='day',data=tips, orient='h')

sns.set_theme(style='whitegrid')

tips['tip_pct'].plot.hist(bins=50)
tips['tip_pct'].plot.density()

comp1 = np.random.normal(0,1,size=200)
comp1 = np.random.randn(200)
comp2 = np.random.normal(10,2,size=200)
comp2 = np.random.randn(200)*2 +10
values = pd.Series(np.concatenate([comp1,comp2]))

sns.distplot(values, bins=100, color='k')


# sns,scatter

mpg = sns.load_dataset('mpg')
mpg.info
# x축은 mpg, y축은 displacement 인 산점도를 그려봅시다
sns.scatterplot(data = mpg, x = 'mpg', y = 'displacement',hue='cylinders')
sns.scatterplot(data = mpg, x = 'mpg', y = 'displacement',hue='origin')

# sns.regplot

sns.regplot(data = mpg, x = 'mpg', y = 'displacement')

# 패싯 그리드 col = 쪼개는 기준 

sns.factorplot(x='day',y='tip_pct',hue='time', col='smoker',
               kind='bar',data=tips[tips.tip_pct < 1])

sns.factorplot(x='day',y='tip_pct',row='time', col='smoker',
               kind='bar',data=tips[tips.tip_pct < 1])

sns.factorplot(x='tip_pct',y='day',
               kind='box',data=tips[tips.tip_pct < 0.5])


sns.catplot(data = tips[tips.tip_pct<0.5])


tips.day
tips.head()

titanic = sns.load_dataset("titanic")

titanic.info()
sns.distplot(titanic['fare'])
sns.distplot(titanic['fare'],hist = False)
sns.distplot(titanic['age'],hist = False)

sns.stripplot(data = titanic, x = 'class', y = 'age')
sns.swarmplot(data = titanic, x = 'class', y = 'age')
sns.violinplot(data = titanic, x = 'class', y = 'age')

sns.jointplot(data = titanic, x = 'fare', y= 'age')
sns.jointplot(data = titanic, x = 'fare', y= 'age',kind='kde')

sns.countplot(data = titanic, )


### 데이터 집계와 그룹 연산

dict1 = {'key1' : ['a','a','b','b','a'],
         'key2': ['one','two','one','two','one'],
         'data1':np.random.randn(5),
         'data2':np.random.randn(5)}
df = pd.DataFrame(dict1)
df

grouped = df['data1'].groupby(df['key1'])
grouped.mean()



for i in grouped:
    print(i)

means = df['data1'].groupby([df['key1'], df['key2']]).mean()
means

means.unstack()

states = np.array(['Ohio','California','California','Ohio','Ohio'])

years = np.array([2005,2005,2006,2005,2006])

df['data1'].groupby([states,years]).mean()

grouped = df.groupby('key1')

for name, group in grouped:
    print(name)
    print(group)

    
grouped = df.groupby(['key1','key2'])

for (k1,k2),group in grouped:
    print((k1,k2))
    print(group)
    
pieces = dict(list(df.groupby('key1')))
pieces['b']

df.dtypes

grouped = df.groupby(df.dtypes, axis=1)

for dtype, group in grouped:
    print(dtype)
    print(group)
    

# 칼럼이나 칼람의 일부만 선택하기

grouped = df.groupby(['key1','key2'])
grouped.sum()
grouped.sum()['data1']
df.groupby(['key1','key2']).sum()['data1']


#0401

df = pd.read_csv("csv_exam.csv")
df[:3]

df.rename(columns = {'class':"classes"},inplace = True)

df.groupby('classes')['english'].mean()

df['total'] = df['english'] + df['math'] + df['science']

sns.barplot(data=df, x = 'classes', y='total')
sns.boxplot(data=df[df['math']>55], x = 'classes', y='math')



# 사전과 Series에서 그루핑 하기

people = pd.DataFrame(np.random.randn(5,5),
                      columns=['a','b','c','d','e'],
                      index=['Joe','Steve','Wes','Jim','Travis'])

people.iloc[2:3,[1,2]]

people

mapping = {'a':'red','b':'red','c':'blue','d':'blue','e':'red','f':'orange'}

by_column = people.groupby(mapping,axis=1)

by_column.sum()

dict1 = {'key1' : ['a','a','b','b','a'],
         'key2': ['one','two','one','two','one'],
         'data1':np.random.randn(5),
         'data2':np.random.randn(5)}
df = pd.DataFrame(dict1)
df

df.groupby('key1').mean()
df.groupby('key1').count()
df.groupby('key1').quantile(0.9)

def peak_to_peak(arr):
    return arr.max() - arr.min()

df.groupby('key1').quantile(0.9).agg(peak_to_peak)

df.describe()
df.groupby('key1').describe()


tips.groupby(['day','smoker']).mean()
tips.groupby(['day','smoker']).agg('mean')
tips.groupby(['day','smoker']).agg(['mean','median'])
tips.groupby(['day','smoker'])['size','tip_pct'].agg(['mean','median'])



tips.groupby(['day','smoker'])

# tip 에는 max size에는 mean
tips.groupby(['day','smoker']).agg({'tip' : 'max','size' : 'mean'})
tips.groupby(['day','smoker'], as_index = False).agg({'tip' : 'max','size' : 'mean'})

df2 = tips.groupby(['day','smoker'], as_index = False).agg({'tip' : 'max','size' : 'mean'})
df2.reset_index()
df2.reset_index(level=0)
tips.groupby(['day','smoker'], as_index = False)


def top(df, n=5, column='tip_pct'):
    return df.sort_values(by=column)[-n:]
def top1(df, n=5, column='tip_pct'):
    return df.sort_values(by=column)[-n:][::-1]

top(tips,6)
top1(tips,6)
top(tips,6,'total_bill')

tips.groupby('smoker').agg(top) # 단일값을 리턴해주는 함수
tips.groupby('smoker').agg('mean')
tips.groupby('smoker').apply(top)
tips.groupby('smoker').apply(np.mean)
tips.groupby('smoker').apply(top, n=2, column = 'total_bill')

frame = pd.DataFrame({'data1':np.random.randn(1000),
                      'data2':np.random.randn(1000)})
frame

quartiles = pd.cut(frame.data1, 4)

grouped = frame.data2.groupby(quartiles)

def get_stats(group):
    return {'min':group.min(),'max':group.max(),'count':group.count(),'mean':group.mean()}

grouped.apply(get_stats).unstack()

tips.pivot_table(values=['size','tip'],index=['day','smoker'], aggfunc ='max')
tips.pivot_table(index=['day','smoker'])
tips.pivot_table(index=['day'], aggfunc='max')
tips.pivot_table(index=['day'],columns=['size'], aggfunc='max')
tips.pivot_table(values = ['tip'],index=['day'],columns=['size'], aggfunc='max')
tips.pivot_table(values = ['tip'],index=['day'],columns=['size'], aggfunc='count')
tips.pivot_table(values = ['tip'],index=['day'],columns=['time'],
                 aggfunc='count',margins=True)
tips.pivot_table(values = ['tip'],index=['day'],columns=['time'],
                 aggfunc=len)
tips.pivot_table(['tip_pct','size'],index=['time','day'],columns='smoker')

pd.crosstab()

tips[['time','day']]
pd.crosstab(tips['time'], tips['day'])
pd.crosstab(tips['time'], tips['smoker'])

tips[:3]

tips.pivot_table(index='day',
                 columns='time',
                 values='total_bill')

tips.pivot_table(index='sex',
                 columns='smoker',
                 values='tip')

tips.pivot_table(index=['day','smoker'],
                 columns='time',
                 values=['total_bill','size'])


dia = sns.load_dataset("diamonds")

sns.scatterplot(data = dia, x = 'carat', y = 'price',hue= 'cut')

dia[:3]
dia.info()

dia.pivot_table(values='price',
                index='cut',
                columns='color').max()
dia.pivot_table(values='price',
                index='cut',
                columns='color',
                aggfunc='max')


# crosstab, contingency table

pd.crosstab(index = dia.cut, columns = dia.color)

dia['cut','color']

# datetime  시계열

from datetime import datetime

now = datetime.now()

now

now.year, now.month, now.day

delta = datetime(2011,1,7) - datetime(2008,6,24,8,15)

delta

delta.days

delta.seconds


# 올해 크리스 마스까지

delta = datetime(2022,12,25) - now

delta.days

datetime.strptime(now, '%Y/%m/%d')

datestrs = now.strftime('%Y-%m-%d')

datetime.strptime(datestrs, '%Y-%m-%d' )

from dateutil.parser import parse

parse('2011-01-03')
parse('2011-1-03')
parse('2011-2-03')
parse('2011-01/03')
parse('03/02/11')

parse("6/12/2011")
parse("6/12/2011", dayfirst = True)


longer_ts = pd.Series(np.random.randn(1000),
                      index=pd.date_range('1/1/2000', periods=1000))

longer_ts1 = pd.Series(np.random.randn(1000),
                      index=pd.date_range('1/1/2000', periods=1000))

pd.concat([longer_ts,longer_ts1],axis = 1)
pd.concat([longer_ts,longer_ts1],axis = 0)

longer_ts['2001-05']
longer_ts['2001-01-05':'2001-01-10']

ts = longer_ts['2000-01']

ts

ts.truncate(after = '1/9/2000')
ts.truncate(before = '1/9/2000')

ts = pd.concat([longer_ts, longer_ts1])

ts['2000-01-05']

# 

ts.groupby(level=0).count()


pd.date_range('2012-04-01','2012-06-01')
pd.date_range('2012-04-01',periods=20)
pd.date_range(end='2012-04-01',periods=20)
pd.date_range('2012-04-01','2012-10-01', freq='BM')
pd.date_range('2012-04-01','2012-10-01', freq='W-TUE')
pd.date_range('2012-04-01',periods=20, freq='2h')
pd.date_range('2012-04-01',periods=20, freq='2h30min')

ts = longer_ts[ :10]

ts.shift(2)
ts.shift(-2)

ts.shift(1)
ts.shift(-1)

ts_p = ts/ts.shift(1) -1

pd.concat([ts,ts_p],axis = 1)


# 문제

tips.groupby(['day','smoker'])['size','tip'].agg(['max'])
tips.groupby(['day','smoker']).agg({'tip' : 'mean','size' : 'max'})

## 0404

gapminder = px.data.gapminder()


df = gapminder[gapminder['year']== 2002]
df = gapminder.query('year == 2002')


ts = pd.Series(np.random.randn(10),
               index=pd.date_range('1/1/2000', periods=10))

df = pd.DataFrame(ts)
df.columns = ['value']
df['value_pct'] = 0
df

for i in range(len(df)-1):
    df.iloc[i+1,1] = df.iloc[i+1,0] / df.iloc[i,0] -1
    

df['value_pct'] = df['value'] / df['value'].shift(1) - 1

import pytz
pytz.common_timezones[:5]

[ i for i in pytz.common_timezones]

##
data = pd.read_csv('macrodata.csv')

data.head()
len(data)

idx = pd.PeriodIndex(year = data['year'],
               quarter = data['quarter'],
               freq='Q-DEC')

data.index = idx

rng = pd.date_range('2000-01-01', periods=100, freq='D')

ts = pd.Series(np.random.randn(len(rng)), index = rng)

ts['2000-01']
ts.resample('M')
list(ts.resample('M'))

ts.resample('M').mean()
ts.resample('M').max()

def peak_to_peak(arr):
    return arr.max() - arr.min()
ts.resample('M').agg(peak_to_peak)

ts.resample('M', kind='period').mean()

rng = pd.date_range('2000-01-01', periods=12, freq='T')
ts = pd.Series(pd.Series(np.arange(12)), index = rng)

ts.resample('5min')
ts.resample('5min').max()
ts.resample('5min').sum()
ts.resample('5min', closed='right').sum()
ts.resample('5min').ohlc()

ts = pd.Series(np.random.randint(100, size = len(rng)), index = rng)


ts['2000-01']


data = pd.read_csv('examples/stock_px_2.csv',
                   parse_dates=True, index_col=0)
data
close_px = data[['AAPL','MSFT','XOM']]

close_px = close_px.resample('B').ffill()
close_px.AAPL.plot()

close_px.AAPL.rolling(250).mean().plot()

appl_std250 = close_px.AAPL.rolling(250, min_periods=10).std()

appl_std250[5:12]

appl_std250.plot()


close_px.rolling(60).mean().plot()
close_px.plot()


# 고급 판다스

dim = pd.Series(['apple','orange'])
values = pd.Series([0,1,0,0]*2)
dim.take(values)

# ipynb -> py
# 1.유틸 ... 명령어
# 2.notebook 에서 py로 저장

fruits = ['apple', 'orange', 'apple', 'apple'] * 2
N = len(fruits)
df = pd.DataFrame({'fruit': fruits,
                   'basket_id': np.arange(N),
                   'count': np.random.randint(3, 15, size=N),
                   'weight': np.random.uniform(0, 4, size=N)},
                  columns=['basket_id', 'fruit', 'count', 'weight'])
df

fruit_cat = df['fruit'].astype('category')
c = fruit_cat.values
type(c)
c.codes
c.categories

cat = pd.Categorical(['foo','bar','baz'])

# cut을 이용해서 만들기

np.random.seed(12345)
draw = np.random.randint(100, size = 100)

pd.qcut(draw, 4)
bins = pd.qcut(draw, 4, labels=['Q1','Q2','Q3','Q4'])
bins.categories
bins.codes

pd.Series(draw).groupby(bins).max
pd.Series(draw).groupby(bins).agg(['max','min','count'])
pd.Series(draw).groupby(bins).agg(['max','min','count']).reset_index

##

label = pd.Series(['foo','bar','baz','qux']*250)
label_categories = label.astype('category')

label.memory_usage()
label_categories.memory_usage()


##

c = pd.Series(['a','b','c','d'], dtype='category')
c = pd.Categorical(['a','b','c','d'], ordered=True)

#
pd.get_dummies(c)



## groupby 심화

df = pd.DataFrame({'key':['a','b','c']*4,
                   'value':np.arange(12)})

df.groupby('key').value
df.groupby('key').max()
df.groupby('key').apply(max)
df.groupby('key').agg(max)
df.groupby('key').transform(max)

df.groupby('key').mean()
df.groupby('key').apply(np.mean)
df.groupby('key').agg(np.mean)
df.groupby('key').transform(np.mean)

g = df.groupby('key')
g.transform(lambda x: x*2)
df.groupby('key').transform(lambda x: x*3)
# normalize 정규화: 표준 정규분포로 바꾸는것, 0~1사이 값으로 조정

def normalize(x):
    return (x - x.mean())/ x.std()

a = pd.Series([1,2,3])    

normalize(a)

df.groupby('key').transform(normalize)
df.groupby('key').apply(normalize)
df.groupby('key').agg(normalize)



# 0405

np.random.randn # 표준 정규 분포

np.random.randint(3)
np.random.randint(1,3)
np.random.randint(1,3,size = 2)
np.random.randint(1,3,size = (2,3))

np.random.uniform(size = 3) # 균등 함수,균등 분포
np.random.uniform(size = 3)+1
np.random.uniform(1,3,size = 3)

np.random.nomrmal(80, 2, 3) # 정규 분포

np.random.randint(3,15,size=7)
np.random.uniform(0,4,size=7)

N = 8
df = pd.DataFrame({'fruit': ['apple', 'orange', 'apple', 'apple'] * 2,

                   'basket_id': np.arange(N),

                   'count': np.random.randint(3, 15, size=N),

                   'weight': np.random.uniform(0, 4, size=N)},

                  columns=['basket_id', 'fruit', 'count', 'weight'])

df
cat = df['fruit'].astype('category')

df['labels'] = pd.qcut(df['weight'],4 , labels = ['Q1','Q2','Q3','Q4'])

df = pd.DataFrame({'key':['a','b','c']*2,
                   'value':np.arange(6)})

df

df.groupby('key').transform(lambda x: x*5)

df.transform(lambda x: x*5)

trs = lambda x: x*5
df['value'] = df['value'].map(lambda x: x*5)


# scikit_learn

pd.read_csv('tatinic.csv')
 
train = pd.read_csv('datasets/titanic/train.csv')
test = pd.read_csv('datasets/titanic/test.csv')

# 결측치 찾기
train.isnull().sum()
test.isnull().sum()

# 결측치 처리하기, 대체
impute_value = train['Age'].median()
train['Age'] = train['Age'].fillna(impute_value)
test['Age'] = test['Age'].fillna(impute_value)

train['IsFemale'] = (train['Sex'] == 'female').astype(int)
test['IsFemale'] = (test['Sex'] == 'female').astype(int)

predictors = ['Pclass','IsFemale','Age']
X_train = train[predictors].values
Y_train = train['Survived'].values
X_test = test[predictors].values

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, Y_train)

# 핏팅, 모델피팅, 적합, 모델적합

y_predict = model.predict(X_test)

y_predict[:10]

## cv = cross validation 교차 검증

from sklearn.linear_model import LogisticRegressionCV

model_cv = LogisticRegressionCV()
model_cv.fit(X_train, Y_train)
y_predict = model_cv.predict(X_test)

from sklearn.model_selection import cross_val_score

model = LogisticRegression()

scores = cross_val_score(model, X_train, Y_train, cv= 4)





##



import pandas as pd

# Make display smaller
pd.options.display.max_rows = 10

unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('datasets/movielens/users.dat', sep='::',
                      header=None, names=unames)

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('datasets/movielens/ratings.dat', sep='::',
                        header=None, names=rnames)

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('datasets/movielens/movies.dat', sep='::',
                       header=None, names=mnames)

users[:5]

ratings[:5]

movies[:5]

ratings

data = pd.merge(pd.merge(ratings, users), movies)
data
data.iloc[0]

mean_ratings = data.pivot_table('rating',index='title',
                               columns='gender',aggfunc='mean')
mean_ratings[:5]

ratings_by_title = data.groupby('title').size()
ratings_by_title[:10]

active_title = ratings_by_title.index[ratings_by_title >= 250]
active_title

mean_ratings = mean_ratings.loc[active_title]
mean_ratings

top_female_ratings = mean_ratings.sort_values(by='F',ascending=False)
top_female_ratings[:10]

mean_ratings['diff'] = mean_ratings['M'] - mean_ratings['F']

sorted_by_diff = mean_ratings.sort_values(by='diff')
sorted_by_diff[::-1][:10]

rating_std_by_title = data.groupby('title')['rating'].std()

rating_std_by_title = rating_std_by_title.loc[active_title]

rating_std_by_title.sort_values(ascending=False)[:10]

data['year'] = data['title'].str[-6:]

data['year'].str.strip('()')



##

fec = pd.read_csv('datasets/fec/P00000001-ALL.csv')

fec.info()

fec.iloc[12345]

unique_cands = fec.cand_nm.unique()
unique_cands
unique_cands[2]

fec.cand_nm
fec.cand_nm.unique()


parties = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}

fec.cand_nm.map(parties)

fec['party'] = fec.cand_nm.map(parties)

fec[['cand_nm','party']]

fec['party'].value_counts()

(fec.contb_receipt_amt > 0 ).value_counts()

fec = fec[fec.contb_receipt_amt > 0]

fec_mrbo = fec[fec.cand_nm.isin(['Obama, Barack','Romney, Mitt'])]


fec.contbr_occupation.value_counts()[:10]

occ_mapping = {
   'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
   'INFORMATION REQUESTED' : 'NOT PROVIDED',
   'INFORMATION REQUESTED (BEST EFFORTS)' : 'NOT PROVIDED',
   'C.E.O.': 'CEO'
}

# If no mapping provided, return x
f = lambda x: occ_mapping.get(x, x)
fec.contbr_occupation = fec.contbr_occupation.map(f)

emp_mapping = {
   'INFORMATION REQUESTED PER BEST EFFORTS' : 'NOT PROVIDED',
   'INFORMATION REQUESTED' : 'NOT PROVIDED',
   'SELF' : 'SELF-EMPLOYED',
   'SELF EMPLOYED' : 'SELF-EMPLOYED',
}

# If no mapping provided, return x
f = lambda x: emp_mapping.get(x, x)
fec.contbr_employer = fec.contbr_employer.map(f)

by_occupation = fec.pivot_table('contb_receipt_amt',
                                index='contbr_occupation',
                                columns='party',
                                aggfunc='sum')

over_2mm = by_occupation[by_occupation.sum(axis = 1) > 2000000]                                
over_2mm

over_2mm.plot(kind='barh')

fec[fec['cand_nm'].isin(['Romney, Mitt','Obama, Barack'])]

def get_top_amounts(group, key, n=5):
    totals = group.groupby(key)['contb_receipt_amt'].sum()
    return totals.nlargest(n)

grouped = fec_mrbo.groupby('cand_nm')
grouped.apply(get_top_amounts, 'contbr_occupation', n = 7)
grouped.apply(get_top_amounts, 'contbr_employer', n = 10)


bins = np.array([0,1,10,100,1000 , 10000, 100000, 1000000, 10000000])

labels = pd.cut(fec_mrbo.contb_receipt_amt, bins)

grouped = fec_mrbo.groupby(['cand_nm', labels])
grouped.size().unstack(0)

data = grouped.size()
data.plot.barh
data.plot.bar()
data.unstack(0)[:3].plot.barh()


data.reset_index().plot.bar()

data.unstack(1).plot.bar
data.unstack(0).plot.bar


bucket_sums = grouped.contb_receipt_amt.sum().unstack(0)
normed_sums = bucket_sums.div(bucket_sums.sum(axis=1), axis = -0)

normed_sums[:-2].plot(kind='barh')

