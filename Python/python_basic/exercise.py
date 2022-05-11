from statistics import mode
import urllib.request as req
import re

rep = req.urlopen("http://www.daum.net")
data = rep.read().decode("utf-8")
print(data)


# 시작은 http:// 끝은 .js

re.findall(".js", data)
re.findall("https://", data)
li = re.findall("h[:./\w]*[.]{1}js", data)

for i in li:
    print(i)


n = 1
sum = 0
while n < 1000:
    if n % 3 == 0:
        sum += n
    elif n % 5 == 0:
        sum += n
    n += 1
print(sum)


result = 0
for n in range(1,1000):
    if n % 3 == 0 or n % 5 == 0:
        result += n
print(result)


def getTotalPage(m,n):
    return m // n + 1

gtp = getTotalPage()

print(gtp(5,10))


import random

max(random.choices(range(1,101),k = 10))

import time

time.strftime("%Hh %Mm %x")

import calendar
import random

a = calendar.prmonth(2022,random.randint(1,12))

import random
random.sample(range(1,11), k=3)


import pickle

f = open("HelloWorld.pickle", mode = "wb")
data = ("HelloWorld")
pickle.dump(data, f)
f.close()

import sys

option = sys.argv[1]

if option == '-a':
    memo = sys.argv[2]
    f = open('memo.txt','a')
    f.write(memo)
    f.write('\n')
    f.close


import re