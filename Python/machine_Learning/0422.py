from selenium import webdriver
import urllib.request as req
from bs4 import BeautifulSoup
import requests
from urllib.parse import quote_plus
import pandas as pd

driver = webdriver.Chrome("C:\python\machine_Learning\chromedriver")
driver.get("https://www.naver.com")


navs = driver.find_elements_by_class_name("nav")

for nav in navs:
    print(nav.text)

from selenium.webdriver.common.keys import Keys
e = driver.find_element_by_id("query")
e.clear()
e.send_keys("python")

e.send_keys(Keys.ENTER)

driver.get("https://www.daum.net/")
driver.find
form = driver.find_element_by_css_selector(".link_login.link_kakaoid")
form.submit()
driver.close()

################################################
import pandas as pd

df = pd.DataFrame(columns=range(2))
title = []
L = []
def serch(x):
    url1 = ("https://search.naver.com/search.naver?where=news&sm=tab_jum&query=")
    url2 = (x)
    url = url1 + url2
    res = req.urlopen(url).read()
    soup = BeautifulSoup(res, 'html.parser')
    links = soup.select("a.news_tit")
    for link in links :
        title.append(link.text)
        L.append(link.attrs['href'])
        
def ques():
    x = quote_plus(input("검색어를 입력해 주세요: "))
    serch(x)

ques()

df = pd.DataFrame({'기사제목': title, '링크' : L})
################
df = pd.DataFrame(columns=("title","href"))

def serch(x):
    url1 = ("https://search.naver.com/search.naver?where=news&sm=tab_jum&query=")
    url2 = (x)
    url = url1 + url2
    res = req.urlopen(url).read()
    soup = BeautifulSoup(res, 'html.parser')
    links = soup.select("a.news_tit")
    count = 0
    for link in links :
        df.loc[count] = [link.text, link.attrs['href']]
        count += 1
        
        
def ques():
    x = quote_plus(input("검색어를 입력해 주세요: "))
    serch(x)

while True:
    ques()
    print(df)
#################################

import requests
import json

apikey = "42b493524ee2ad40c03c6f5359d366de"

cities = ["Seoul,KR", "Tokyo,JP", "New York,US"]

api = "http://api.openweathermap.org/data/2.5/weather?q={city}&APPID={key}"

k2c = lambda k: k - 273.15

# 각 도시의 정보 추출하기
for name in cities:
    # API의 URL 구성하기
    url = api.format(city=name, key=apikey)
    # API에 요청을 보내 데이터 추출하기
    r = requests.get(url)
    # 결과를 JSON 형식으로 전환하기
    data = json.loads(r.text)
    # 결과 출력하기
    print("+ 도시 =", data["name"])
    print("| 날씨 =", data["weather"][0]["description"])
    print("| 최저 기온 =", k2c(data["main"]["temp_min"]))
    print("| 최고 기온 =", k2c(data["main"]["temp_max"]))
    print("| 습도 =", data["main"]["humidity"])
    print("| 기압 =", data["main"]["pressure"])
    print("| 풍향 =", data["wind"]["deg"])
    print("| 풍속 =", data["wind"]["speed"])
    print("")

#############################################
from bs4 import BeautifulSoup
import urllib.request as req
import datetime
# HTML 가져오기
url = "http://finance.naver.com/marketindex/"
res = req.urlopen(url)
# HTML 분석하기
soup = BeautifulSoup(res, "html.parser")
# 원하는 데이터 추출하기 --- (※1)
price = soup.select_one("div.head_info > span.value").string
print("usd/krw", price)
# 저장할 파일 이름 구하기
t = datetime.date.today()
fname = t.strftime("%Y-%m-%d") + ".txt"
with open(fname, "w", encoding="utf-8") as f:
    f.write(price)

#################################################    
filename = "a.bin"
data = 100
# 쓰기
with open(filename, "wb") as f:
    f.write(bytearray([data]))

filename = "a.bin"
data = 97
# 쓰기
with open(filename, "wb") as f:
    f.write(bytearray([data]))

##################################################
df.to_csv("link1.csv",encoding="cp949")

#################################################
from bs4 import BeautifulSoup 
import urllib.request as req
import os.path


url = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108"
savename = "forecast.xml"

if not os.path.exists(savename):
    req.urlretrieve(url, savename)

# BeautifulSoup로 분석하기 --- (※2)
xml = open(savename, "r", encoding="utf-8").read()
soup = BeautifulSoup(xml, 'html.parser')

# 각 지역 확인하기 --- (※3)
info = {}
for location in soup.find_all("location"):
    name = location.find('city').string
    weather = location.find('wf').string
    if not (weather in info):
        info[weather] = []
    info[weather].append(name)

# 각 지역의 날씨를 구분해서 출력하기
for weather in info.keys():
    print("+", weather)
    for name in info[weather]:
        print("| - ", name)

#############################################
import urllib.request as req
import os.path, random
import json


# JSON 데이터 내려받기 --- (※1)
url = "https://api.github.com/repositories"
savename = "repo.json"

if not os.path.exists(savename):
    req.urlretrieve(url, savename)

# JSON 파일 분석하기 --- (※2)
items = json.load(open(savename, "r", encoding="utf-8"))
# 또는
# s = open(savename, "r", encoding="utf-8").read()
# items = json.loads(s)
# 출력하기 --- (※3)
for item in items:
    print(item["name"] + " - " + item["owner"]["login"])

#################################################
import codecs
# EUC_KR로 저장된 CSV 파일 읽기
filename = "C:/python/machine_Learning/pyml-rev-main/ch3/list-euckr.csv"
csv = codecs.open(filename, "r", "euc_kr").read()

# CSV을 파이썬 리스트로 변환하기
data = []
rows = csv.split("\r\n")

for row in rows:
    if row == "": continue
    cells = row.split(",")
    data.append(cells)

# 결과 출력하기
for c in data:
    print(c[1], c[2])

###############################################
import csv, codecs


# CSV 파일 열기
filename = "C:/python/machine_Learning/pyml-rev-main/ch3/list-euckr.csv"
fp = codecs.open(filename, "r", "euc_kr")

# 한 줄씩 읽어 들이기
reader = csv.reader(fp, delimiter=",", quotechar='"')

for cells in reader:
    print(cells[1], cells[2])

###############################################

import pandas as pd
pd.read_csv("test.csv", encoding="euc-kr")
pd.read_csv("test.csv", encoding="cp949")

##################################################

import openpyxl 


# 엑셀 파일 열기く --- (※1)
filename = "stats_104102.xlsx"
book = openpyxl.load_workbook(filename)

# 맨 앞의 시트 추출하기 --- (※2)
sheet = book.worksheets[0]

# 시트의 각 행을 순서대로 추출하기 --- (※3)
data = []
for row in sheet.rows:
    data.append([
        row[0].value,
        row[10].value
    ])

# 필요없는 줄(헤더, 연도, 계) 제거하기
del data[0]
del data[1]
del data[2]

# 데이터를 인구 순서로 정렬합니다.
data = sorted(data, key=lambda x:x[1])

# 하위 5위를 출력합니다.
for i, a in enumerate(data):
    if (i >= 5): break
    print(i+1, a[0], int(a[1]))


#############

df = pd.read_excel(filename, header = None)

df.reindex()

df.reset_index(drop = True)


df.iloc[:, [0,10]]

df.sort_values(by = [10])[:5]


df.iloc[:, [0,10]].sort_values(by=[10], axis=0).head()

df = df.drop(columns = range(1, 10)) 
df.sort_values(by = [10])[:5]

################################

import openpyxl 


# 엑셀 파일 열기 --- (※1)
filename = "stats_104102.xlsx"
book = openpyxl.load_workbook(filename)

# 활성화된 시트 추출하기 --- (※2)
sheet = book.active

# 서울을 제외한 인구를 구해서 쓰기 --- (※3)
for i in range(0, 10):
    total = int(sheet[str(chr(i + 66)) + "3"].value)
    seoul = int(sheet[str(chr(i + 66)) + "4"].value)
    output = total - seoul
    print("서울 제외 인구 =", output)
    # 쓰기 --- (※4)
    sheet[str(chr(i + 66)) + "21"] = output
    cell = sheet[str(chr(i + 66)) + "21"]
    # 폰트와 색상 변경해보기 --- (※5)
    cell.font = openpyxl.styles.Font(size=14,color="FF0000")
    cell.number_format = cell.number_format

# 엑셀 파일 저장하기 --- (※6)
filename = "population.xlsx"
book.save(filename)
print("ok")

#################################################
import sqlite3


# sqlite 데이터베이스 연결하기 --- (※1)
dbpath = "test.sqlite"
conn = sqlite3.connect(dbpath)

# 테이블을 생성하고 데이터 넣기 --- (※2)
cur = conn.cursor()
cur.executescript("""
/* items 테이블이 이미 있다면 제거하기 */
DROP TABLE IF EXISTS items;
/* 테이블 생성하기 */
CREATE TABLE items(
    item_id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    price INTEGER
);
/* 데이터 넣기 */
INSERT INTO items(name, price)VALUES('Apple', 800);
INSERT INTO items(name, price)VALUES('Orange', 780);  
INSERT INTO items(name, price)VALUES('Banana', 430);
""")

# 위의 조작을 데이터베이스에 반영하기 --- (※3)
conn.commit()

# 데이터 추출하기 --- (※4)
cur = conn.cursor()
cur.execute("SELECT item_id,name,price FROM items")
item_list = cur.fetchall()

# 출력하기
for it in item_list:
    print(it)

conn.commit()

cur = conn.cursor()
cur.execute("SELECT item_id, name, price FROM items")
item_list = cur.fetchall()

for it in item_list:
    print(it)
    
cur.execute("SELECT * FROM items WHERE name = 'Apple'")
item_list = cur.fetchall()
print(item_list)

