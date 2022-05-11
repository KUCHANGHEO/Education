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
        print(link.text)
        print(link.attrs['href'])
        print("\n")

def ques():
    x = quote_plus(input("검색어를 입력해 주세요: "))
    serch(x)
    
while True:
    ques()
