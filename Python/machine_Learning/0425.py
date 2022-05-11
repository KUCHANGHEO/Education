from selenium import webdriver
import urllib.request as req
from bs4 import BeautifulSoup
import requests
from urllib.parse import quote_plus
import pandas as pd
import json
import datetime
import os.path, random
import csv, codecs
import openpyxl 
import sqlite3
import cx_Oracle
import sklearn
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import random, re
import numpy as np
# 연습

with open('C:/python/machine_Learning/answer.txt', 'r') as f:
    data = f.readlines()

print(data)
column_name = ['문제번호','정답']
data_split = [x.strip().split() for x in data[0:]]
df1 = pd.DataFrame(data_split, columns= column_name)
df2 = pd.read_clipboard()
df2.columns = column_name
df2 = df2.astype('object')
df1 = df1.astype('object')

df1.iloc[:,1]

df = pd.concat([df1,df2],axis=1)

mark = 0
for i in range(len(df)):
    if df.iloc[i, 1] == df.iloc[i, 3]:
        print(f"문제 {i+1}번 정답")
        mark += 4
    else:
        print(f"문제 {i+1}번 오답")
print(f"총점: {mark}")


(df.iloc[:, 1] == df.iloc[:, 3]).sum() *4


###################################################

dsn = cx_Oracle.makedsn('localhost', 1521, 'orcl')
conn = cx_Oracle.connect('MADANG','MADANG',dsn)

cursor = conn.cursor()

cursor.execute("select * from BOOK")

x = cursor.fetchall()


df_oracle = pd.DataFrame(x)

####

cursor = conn.cursor()

cursor.execute("insert into BOOK(BOOKID, BOOKNAME, PUBLISHER) VALUES (13, '스포츠 의학3', '한솔의학서적')")
    
########################################

from sklearn import svm

xor_data = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
    ]

data = []
label = []
for row in xor_data:
    p = row[0]
    q = row[1]
    r = row[2]
    data.append([p,q])
    label.append(r)

clf = svm.SVC()
clf.fit(data,label)

pre = clf.predict(data)
print("예측결과:",pre)

ok = 0; total = 0
for idx, answer in enumerate(label):
    p = pre[idx]
    if p == answer: ok += 1
    total += 1
print("정답률:", ok, "/", total, "=", ok/total)

###############################################

df = pd.DataFrame(xor_data)

data = df.iloc[:, :2]
label = df.iloc[:, 2:]

# 모델 선택
clf = svm.SVC()

# 학습하기
clf.fit(data, label)

# 예측하기
y_pred = clf.predict(data)

# 평가하기
label.T == y_pred
label.T == y_pred.sum()/4

#############################################
from sklearn import svm, metrics

xor_input = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
    ]

xor_df = pd.DataFrame(xor_input)
xor_data = xor_df.loc[:,0:1]
xor_label = xor_df.loc[:,2]

# 모델선택
clf = svm.SVC()
# 학습하기
clf.fit(xor_data, xor_label)
# 예측하기
pre = clf.predict(xor_data)

ac_score = metrics.accuracy_score(xor_label, pre)
print("정답률 =", ac_score)

#################################################

import random, re

csv = []

with open('C:/python/machine_Learning/iris.csv', 'r', encoding='utf-8') as fp:
    for line in fp:
        line = line.strip()
        cols = line.split(',')
        fn = lambda n : float(n) if re.match(r'^[0-9\.]+$', n) else n
        cols = list(map(fn, cols))
        csv.append(cols)
    
del csv[0]

random.shuffle(csv)

total_len = len(csv)
train_len = int(total_len * 2 / 3)
train_data = []
train_label = []
test_data = []
test_label = []

for i in range(total_len):
    data = csv[i][0:3]
    label = csv[i][4]
    if i < train_len:
        train_data.append(data)
        train_label.append(label)
    else:
        test_data.append(data)
        test_label.append(label)

clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

pd.DataFrame(test_label).values.T == pre
(np.array(test_label).T == pre).sum()
 
(label.values.T == pre).sum()/4

as_score = metrics.accuracy_score(test_label, pre)
print("정답률 =", ac_score)

##################################################

iris = pd.read_csv("C:/python/machine_Learning/iris.csv")

train_index = random.sample(range(150), 100)
text_index = np.setdiff1d(np.arange(150), train_index,assume_unique=True) 
len(train_index)

train_data = iris.iloc[ train_index ,:4]
train_label = iris.iloc[ train_index ,4:]
test_data = iris.iloc[ text_index ,:4]
test_label = iris.iloc[ text_index ,4:]

# 모델선택
clf = svm.SVC()
# 학습하기
clf.fit(train_data, train_label)
# 예측하기
y_pre = clf.predict(test_data)

# 몇 개
metrics.accuracy_score(test_label, y_pre)

# 함수 이용
train, test = train_test_split(iris, test_size= int(len(iris)/3))
len(train)
len(test)


data = iris.iloc[:,:4]
label = iris.iloc[:,4:]
train_data, test_data, train_label, test_label = train_test_split( data, label, test_size= int(len(iris)/3))

##############################################
from sklearn.model_selection import train_test_split

csv = pd.read_csv("C:/python/machine_Learning/iris.csv")

csv_data = csv[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
csv_label = csv["Name"]

train_data, test_data, train_label, test_label = \
    train_test_split(csv_data, csv_label)


clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

as_score = metrics.accuracy_score(test_label, pre)
print("정답률 =", ac_score)

#################################################

import urllib.request as req
import gzip, os, os.path

savepath = "./mnist"
baseurl = "http://yann.lecun.com/exdb/mnist"
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"]

# 다운로드
if not os.path.exists(savepath): os.mkdir(savepath)
for f in files:
    url = baseurl + "/" + f
    loc = savepath + "/" + f
    print("download:",url)
    if not os.path.exists(loc):
        req.urlretrieve(url, loc)

for f in files:
    gz_file = savepath + "/" + f
    raw_file = savepath + "/" + f.replace(".gz","")
    print("gzip:", f)
    with gzip.open(gz_file, "rb") as fp:
        body = fp.read()
        with open(raw_file, "wb") as w:
            w.write(body)
print("ok")

############################

import struct

def to_csv(name, maxdata):
    # 레이블 파일과 이미지 파일 열기
    lbl_f = open("./mnist/"+name+"-labels-idx1-ubyte", "rb")
    img_f = open("./mnist/"+name+"-images-idx3-ubyte", "rb")
    csv_f = open("./mnist/"+name+".csv","w", encoding="utf-8")
    # 헤더 정보 읽기
    mag, lbl_count = struct.unpack(">II", lbl_f.read(8))
    mag, img_count = struct.unpack(">II", img_f.read(8))
    rows, cols = struct.unpack(">II", img_f.read(8))
    pixels = rows * cols
    # 이미지 데이터를 읽고 CSV로 저장하기
    for idx in range(lbl_count):
        if idx > maxdata: break
        label = struct.unpack("B", lbl_f.read(1))[0]
        bdata = img_f.read(pixels)
        sdata = list(map(lambda n: str(n), bdata))
        csv_f.write(str(label)+",")
        csv_f.write(",".join(sdata)+"\r\n")
        # 잘 저장됐는지 이미지 파일로 저장해서 테스트 하기
        if idx < 10:
            s = "P2 28 28 255\n"
            s += " " .join(sdata)
            iname = "./mnist/{0}-{1}-{2}.pgm".format(name, idx,label)
        with open(iname, "w", encoding="utf-8") as f:
            f.write(s)
    csv_f.close()
    lbl_f.close()
    img_f.close()

to_csv("train", 1000)
to_csv("t10k", 500)


############################################

from sklearn import model_selection, svm, metrics

# CSV 파일을 읽고 들이고 가공하기
def load_csv(fname):
    labels = []
    images = []
    with open(fname, "r") as f:
        for line in f:
            cols = line.split(",")
            if len(cols) < 2: continue
            labels.append(int(cols.pop(0)))
            vals = list(map(lambda n: int(n) / 256, cols))
            images.append(vals)
    return {"labels":labels, "images":images}

data = load_csv("./mnist/train.csv")
test = load_csv("./mnist/t10k.csv")

# 학습하기
clf = svm.SVC()
clf.fit(data["images"],data["labels"])

# 예측하기
predict = clf.predict(test["images"])

# 결과 확인하기
as_score = metrics.accuracy_score(test["labels"], predict)
cl_report = metrics.classification_report(test["labels"], predict)

print("정답률 =", as_score)
print("리포트 =")
print(cl_report)

###########################

#6만개
to_csv("train", 99999)
to_csv("t10k", 500)
