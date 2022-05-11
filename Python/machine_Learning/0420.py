# bs-test1.py
import urllib.request
import urllib.parse

from bs4 import BeautifulSoup

# 분석하고 싶은 HTML
html = """
<html><body>
  <h1>스크레이핑이란?</h1>
  <p>웹 페이지를 분석하는 것</p>
  <p>원하는 부분을 추출하는 것</>
</body><html>
"""

# HTML 분석하기
soup = BeautifulSoup(html, 'html.parser')

# 원하는 부분 추출하기
h1 = soup.html.body.h1
p1 = soup.html.body.p
p2 = p1.next_sibling.next_sibling

# 요소의 글자 출력하기
print("h1 = " ,h1.string)
print("p = " ,p1.string)
print("p = " ,p2.string)

###############################
# bs-test2.py

from bs4 import BeautifulSoup

# 분석하고 싶은 HTML
html = """
<html><body>
  <h1 id ="title">스크레이핑이란?</h1>
  <p id="body">웹 페이지를 분석하는 것</p>
  <p>원하는 부분을 추출하는 것</>
</body><html>
"""

# HTML 분석하기
soup = BeautifulSoup(html, 'html.parser')

# find() 메소드로 원하는 부분 추출하기
title = soup.find(id="title")
body = soup.find(id="body")

# 텍스트 부분 출력하기
print("#title=" + title.string)
print("#body=" + body.string)


##################################
# bs-link.py

from bs4 import BeautifulSoup

html= """
<html>
 <body>
  <ul>
   <li><a href="http://www.naver.com">naver</a></li>
   <li><a href="http://www.daum.net">daum</a></li>
  </ul>
 </body>
</html>
"""

# html 분석
soup = BeautifulSoup(html, "html.parser")

# find_all() 메소드로 추출하기
links = soup.find_all("a")

# 링크 목록 출력하기
for a in links:
    href = a.attrs['href']
    text = a.string
    print(text, ">", href)
    
########################################
# bs-forcast.py

from bs4 import BeautifulSoup
import urllib.request as req

url = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"

# urlopen() 으로 데이터 가져오기
res = req.urlopen(url)

# BeautifulSoup으로 분석하기
soup = BeautifulSoup(res, "html.parser")

# 원하는 데이터 추출하기
title =  soup.find("title").string
wf = soup.find("wf").string
print(title)
print(wf)

#######################################
# bs-select.py

from bs4 import BeautifulSoup

# 분석 대상 HTML

html = """
<html>
 <body>
  <div id="meigen">
   <h1>위키북스 도서</h1>
   <ul class="item">
    <li>유니티 게임 이펙트 입문</li>
    <li>스위프트로 시작하는 아이폰 앱 개발 교과서</li>
    <li>모던 웹사이트 디자인의 정석</li>
   </ul>
  </div>
 </body>
</html>
"""

# HTML 분석하기
soup = BeautifulSoup(html, "html.parser")

# 필요한 부분을 CSS 쿼리로 추출하기
# 타이틀 부분 추출하기
h1 = soup.select_one("div#meigen > h1").string
print("h1 = ", h1)
# 목록 부분 추출하기
li_list = soup.select("div#meigen > ul.item > li")
for li in li_list:
    print("li=", li.string)
    
#################################
# bs-usd.py

from bs4 import BeautifulSoup
import urllib.request as req

# HTML 가져오기
url = "http://finance.naver.com/marketindex/"
res = req.urlopen(url)

# HTML 분석하기
soup = BeautifulSoup(res, "html.parser")

# 원하는 데이터 추출하기
price = soup.select_one("div.head_info > span.value").string
print("usd/krw =", price)
soup.select("div.head_info.point_dn > span.value")[0].string
soup.find("span.values")
soup.find_all("span",class_='value')[9].text
soup.find_all("span",class_='value')[9].text
soup.select_one("div.market3 > div.data > ul > li:nth-of-type(2) > a.head.gasoline > div.head_info > span.value").string

########################################
import urllib.request as req
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

def serch(x):
    url1 = ("https://search.naver.com/search.naver?where=news&sm=tab_jum&query=")
    url2 = (x)
    url = url1 + url2


    res = req.urlopen(url).read()
    soup = BeautifulSoup(res, 'html.parser')
    links = soup.select("a.news_tit")
    for link in links :
        ##print(f"{link.text}\n{link.attrs['href']}\n\n")
        print(link.text)
        print(link.attrs['href'])
        print("\n")

def ques():
    x = quote_plus(input("검색어를 입력해 주세요: "))
    serch(x)
    
while True:
    ques()

#######################################
# sel-dongju.py

from bs4 import BeautifulSoup
import urllib.request as req

url = "https://ko.wikisource.org/wiki/%EC%A0%80%EC%9E%90:%EC%9C%A4%EB%8F%99%EC%A3%BC"
res = req.urlopen(url)
soup = BeautifulSoup(res, "html.parser")

# #mw-content-text 바로 아래에 있는
# ul 태그 바로 아래에 있는
# li 태그 아래에 있는
# a 태그를 모두 선택합니

a_list = soup.select("#mw-content-text > div > ul > li a")

for a in a_list:
    name = a.string
    print("-", name)
    

######################
# sel-books.py
from bs4 import BeautifulSoup
fp = open("C:/python/machine_Learning/pyml-rev-main/ch1/books.html", encoding="utf-8")
soup = BeautifulSoup(fp, "html.parser")

# CSS 선택자로 검색하는 방법
sel = lambda q : print(soup.select_one(q).string)
sel("#nu")
sel("li#nu")
sel("ul > li#nu")
sel("#bible #nu")
sel("#bible > #nu")
sel("ul#bible > #nu")
sel("li[id='nu']")
sel("li:nth-of-type(4)")

# 그 밖의 방법
print(soup.select("li")[3].string)
print(soup.find_all("li")[3].string)

########################################
# sel-avocado.py

from bs4 import BeautifulSoup
fp = open("C:/python/machine_Learning/pyml-rev-main/ch1/fruits-vegetables.html", encoding="utf-8")
fb ='''
<html>
<body>
<div id="main-goods" role="page">
  <h1>과일과 야채</h1>
  <ul id="fr-list">
    <li class="red green" data-lo="ko">사과</li>
    <li class="purple" data-lo="us">포도</li>
    <li class="yellow" data-lo="us">레몬</li>
    <li class="yellow" data-lo="ko">오렌지</li>
  </ul>
  <ul id="ve-list">
    <li class="white green" data-lo="ko">무</li>
    <li class="red green" data-lo="us">파프리카</li>
    <li class="black" data-lo="ko">가지</li>
    <li class="black" data-lo="us">아보카도</li>
    <li class="white" data-lo="cn">연근</li>
  </ul>
</div>
</body>
</html>
'''

soup = BeautifulSoup(fb, "html.parser")

# CSS 선택자로 추출하기
print(soup.select_one("li:nth-of-type(8)").string)
print(soup.select_one("#ve-list > li:nth-of-type(4)").string)
print(soup.select_one("#ve-list > li[data-lo='us']")[1].string)
print(soup.select_one("#ve-list > li.black")[1].string)

# find 메소드로 추출하기
cond = {"data-lo":"us", "class":"black"}
print(soup.find("li", cond).string)

# find 메소드를 연속적으로 사용하기
print(soup.find(id="ve-list").find("li", cond).string)

########################################
# sel-re.py
from bs4 import BeautifulSoup 
import re # 정규 표현식을 사용할 때 --- (※1)
html = """
<ul>
  <li><a href="hoge.html">hoge</li>
  <li><a href="https://example.com/fuga">fuga*</li>
  <li><a href="https://example.com/foo">foo*</li>
  <li><a href="http://example.com/aaa">aaa</li>
</ul>
"""
soup = BeautifulSoup(html, "html.parser")
# 정규 표현식으로 href에서 https인 것 추출하기 --- (※2)
li = soup.find_all(href=re.compile(r"^https://"))
for e in li: print(e.attrs['href'])
