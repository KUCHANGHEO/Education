# download-png1.py

# 라이버러리 읽어 들이기
import urllib.request

# URL과 저장 경로 지정하기
url = "http://uta.pw/shodou/img/28/214.png"
savename = "test.png"

# 다운로드
urllib.request.urlretrieve(url, savename)
print("저장되었습니다...!")

############################
# download-png2.py

# 라이브러리
import urllib.request

# URL과 저장경로 지정
url = "http://uta.pw/shodou/img/28/214.png"
savename = "test.png"

# 다운로드
mem = urllib.request.urlopen(url).read()

# 파일로 저장하기
with open(savename, mode="wb") as f:
    f.write(mem)
    print("저장되었습니다")
    
#############################
# download-ip.py

# 모듈 읽기
import urllib.request

# 데이터 읽어 들이기
url = "http://api.aoikujira.com/ip/ini"
res = urllib.request.urlopen(url)
data = res.read()

# 바이너리 문자열로 변환하기
text = data.decode("utf-8")
print(text)


###############################
# download-forecast.py

import urllib.request
import urllib.parse

API = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"
# 매개변수를 URL 인코딩
values = {'stnId': '109'}

params = urllib.parse.urlencode(values)
# 요청 전용 URL을 생성
url = API + "?" + params
print("url=", url)
# 다운로드 합니다.
data = urllib.request.urlopen(url).read()
text = data.decode("utf-8")
print(text)

##################
# download-forecast-argv.py
import urllib.request
import urllib.parse
import sys

API = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"

# 명령줄 매개변수 추출
regionNumber = sys.argv[1]
# 매개변수를 URL인코딩
values = {'stnId': regionNumber}

params = urllib.parse.urlencode(values)

url = API + "?" + params
print("url=", url)

data = urllib.request.urlopen(url).read()

text = data.decode("utf-8")
print(text)


########################

