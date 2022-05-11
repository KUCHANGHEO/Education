import urllib.request
import urllib.parse
import sys

API = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"

regionNumber = sys.argv[1]
values = {'stnId': regionNumber}

params = urllib.parse.urlencode(values)

url = API + "?" + params
print("url=", url)

data = urllib.request.urlopen(url).read()

text = data.decode("utf-8")
print(text)