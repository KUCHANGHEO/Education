import urllib.request as req
import urllib.parse
from bs4 import BeautifulSoup

html = """

<!DOCTYPE html>
    <body>
        <h1>ABC</h1>
        <div id="hangle">가나다</div>
        <ul>
            <li class="one">1번</li>
            <li class="two">2번</li>
        </ul>	
    </body>
</html>

"""

soup = BeautifulSoup(html, 'lxml')

soup.select("h1")[0].text

soup.find("div").string

url = ("https://ko.wikipedia.org/wiki/%EB%8C%80%ED%95%9C%EB%AF%BC%EA%B5%AD%EC%9D%98_%EB%8C%80%ED%86%B5%EB%A0%B9_%EB%AA%A9%EB%A1%9D")

res = req.urlopen(url)

soup = BeautifulSoup(res, "lxml")


bigs = soup.select("big > a")

for big in bigs :
    print(big.text)

soup.select("big > a")
###################################################
# user agent 사용하기
import requests

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
    
for i in range(1,10):
    url = "https://search.daum.net/search?w=news&DA=PGD&enc=utf8&cluster=y&cluster_page=1&q=java&p={}".format(i)
    html = requests.get(url, headers = headers).text
    soup = BeautifulSoup(html, "lxml")
    info = soup.find_all(class_ = "tit_main fn_tit_u")
    # print(infos)

    for a in info:
        link = a.attrs['href']
        text = a.text
        print(text, link)
        print()

##################################################
# 이미지 저장하기
import numpy as np
import cv2
url = ("https://search.naver.com/search.naver?where=image&sm=tab_jum&query=%ED%99%8D%EA%B8%B8%EB%8F%99")
res = req.urlopen(url)
soup = BeautifulSoup(res, "lxml")
#main_pack > section.sc_new.sp_nimage._prs_img._imageSearchPC > div > div.photo_group._listGrid > div.photo_tile._grid > div:nth-child(1) > div > div.thumb > a > img
img = soup.find_all("img")

for i in img:
    print(i.attrs['src'])
    
    
######## 셀레니움 사용하기

# 1. urllib.request
# 2. urlopen().read()
# 3. user-agent
# 4. selenium

from selenium import webdriver
import time

driver = webdriver.Chrome("C:\python\machine_Learning\chromedriver")
url = ("https://search.naver.com/search.naver?where=image&sm=tab_jum&query=%ED%99%8D%EA%B8%B8%EB%8F%99")
driver.get(url)
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')
images = soup.find_all(class_="_image _listImage")

count = 0
for image in images:
    image_url = image.attrs['data-lazy-src']
    count += 1
    with req.urlopen(image_url) as f:
        img = f.read()
        with open(f"C:/python/hong/{count}.jpg", 'wb') as g:
            g.write(img)
            
time.sleep(2)
driver.close()

###############################
# login-getmileage.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

USER = "wjsrn17"
PASS = "rn134679@"

session = requests.session()

login_info = {
    "m_id" : USER,
    "m_passwd" : PASS
}

url_login = "https://www.hanbit.co.kr/member/login_proc.php"
res = session.post(url_login, data = login_info)
res.raise_for_status()

url_mypage = "https://www.hanbit.co.kr/myhanbit/myhanbit.html"
res = session.get(url_mypage)
res.raise_for_status()

soup = BeautifulSoup(res.text, "html.parser")
mileage = soup.select_one(".mileage_section1 span").get_text()
ecoin = soup.select_one(".mileage_section2 span").get_text()
print("마일리지: " + mileage)
print("이코인: " + ecoin)


###############################################

# 데이터 가져오기

import requests
r = requests.get("http://api.aoikujira.com/time/get.php")

text = r.text

print(text)

bin = r.content
print(bin)

# 이미지 데이터 추출하기
import requests
r = requests.get("http://uta.pw/shodou/img/28/214.png")
with open("test2.png", "wb") as f:
    f.write(r.content)
print("saved")
urllib.request.urlretrieve("http://uta.pw/shodou/img/28/214.png", "test1.png")

####################################################
# selenium

driver = webdriver.Chrome("C:\python\machine_Learning\chromedriver")
url = ("https://www.naver.com/")
driver.get(url)
driver.save_screenshot("Website.png")
driver.close()

##################################################
# 네이버 로그인
from selenium import webdriver

USER = "wjsrn17"
PASS = "Rnckd134679@"

driver = webdriver.Chrome("C:\python\machine_Learning\chromedriver")
driver.get("https://nid.naver.com/nidlogin.login?mode=form&url=https%3A%2F%2Fwww.naver.com")

e = driver.find_element_by_id("id")
e.clear()
e.send_keys(USER)

e = driver.find_element_by_id("pw")
e.clear()
e.send_keys(PASS)

from selenium.webdriver.common.keys import Keys
e.send_keys(Keys.ENTER)
#form = driver.find_element_by_css_selector("input.btn_login[type=submit]")

driver.get("https://order.pay.naver.com/home?yabMenu=SHOPPING")

products = driver.find_elements_by_class_name("goods")
print(products)

for product in products:
    print("-", product.text)

driver.close()

#######################################

from selenium import webdriver

USER = "wjsrn17"
PASS = "rn134679@"

driver = webdriver.Chrome("C:\python\machine_Learning\chromedriver")
driver.get("https://www.hanbit.co.kr/member/login.html")

e = driver.find_element_by_id("m_id")
e.clear()
e.send_keys(USER)

e = driver.find_element_by_id("m_passwd")
e.clear()
e.send_keys(PASS)

from selenium.webdriver.common.keys import Keys
e.send_keys(Keys.ENTER)

driver.get("https://www.hanbit.co.kr/myhanbit/myhanbit.html")

mileage = driver.find_element_by_css_selector("dd")

mileage.text

driver.close()
