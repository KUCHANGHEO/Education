library(dplyr)
library(ggplot2)

plot(mpg)
plot(mpg$displ, mpg$year)

data <- read.table("https://raw.githubusercontent.com/holtzy/data_to_viz/master/Example_dataset/1_OneNum.csv", header=TRUE)

# ggplot(data=데이터 셋명): 데이터를 불러오는 역활
# mpping=aes(x=,y=): x출,y축 설정
# geom_function(): 어떤 그래프를 그릴지 설정
# positon(x,y),color(색상),fill(채우기),shape(모양)
# linetype(선 형태),size(크기) 등

# Make the histogram
data %>%
  filter( price<300 ) %>%
  ggplot( aes(x=price)) +
  geom_density(fill="#69b3a2", color="#e9ecef", alpha=0.8) 

# 산점도 그리기
mpg
ggplot(data = mpg, aes(x = displ, y= hwy)) +
  geom_point() +
  xlim(3,6)+ # x축 범위 3~6으로 지정
  ylim(20,30) # y축 범위 10~30으로 지정

ggplot(data = mpg, aes(x = displ, y = hwy))+geom_point() #num, num
ggplot(data = mpg, aes(x = displ, y = cty))+geom_point() #num, num
ggplot(data = mpg, aes(x = displ, y = fl))+geom_point() #num, char
ggplot(data = mpg, aes(x = drv, y = fl))+geom_point() #char, char

str(mpg)
glimpse(mpg)
mpg %>% select(displ,hwy, cty)
ggplot(data = mpg, aes(x = displ, y= hwy, color = drv )) + geom_point() # 색상이 범주형
ggplot(data = mpg, aes(x = displ, y= hwy, color = model )) + geom_point() # 색상이 범주형
ggplot(data = mpg, aes(x = displ, y= hwy, color = fl )) + geom_point() #  색상이 범주형
ggplot(data = mpg, aes(x = displ, y= hwy, color = year )) + geom_point() #색상이 연속형(int)

# ggplot뿐아니라 geom_function에 넣어줘도 된다
ggplot(data = mpg, aes(x = displ, y= hwy, color = drv )) + 
  geom_point(size = 3) 
ggplot(data = mpg, aes(x = displ, y= hwy, color = drv )) + 
  geom_point(size = 3, color = "blue") 
ggplot(data = mpg, aes(x = displ, y= hwy, color = drv )) + 
  geom_point(size = 3, aes(color = "drv"))
# mpg에서 파이프 형태로도 가능 )주의 data를 설정 안해도 됨
mpg %>% ggplot()+ geom_point(size = 2, aes(x = displ, y= hwy, color = drv))


# 코드 재활용이 쉽도록

p <-  ggplot(data = mpg , aes(x=displ, y=hwy, color = drv))
p
q <- geom_point(size = 2)
p + q

# bar 그래프는 변수가 x한개여야 됨
ggplot(data = mpg, aes(x = displ, color = drv )) + geom_bar()
# color는 테두리
ggplot(data = mpg, aes(x = drv, color = drv )) + geom_bar()
# fill은 채우기
ggplot(data = mpg, aes(x = drv, fill = drv )) + geom_bar()
# position("dodge")는 중첩된 그래프를 분리해서 보여준다
ggplot(data = mpg, aes(x = drv, fill = fl )) + geom_bar(position = "dodge")
mpg %>% select(drv)
mpg$drv
names(mpg)
ggplot(data = mpg, aes(x = drv)) + geom_bar()
ggplot(data = mpg, aes(x = model)) + geom_bar()
ggplot(data = mpg, aes(x = cty)) + geom_bar()
ggplot(data = mpg, aes(x = year)) + geom_bar()
mpg$drv
# shape는 모양을 바꾼다
ggplot(mpg, aes(x = cty, y = hwy)) +
  geom_point(alpha = 0.9, color = "blue", size = 3, shape = 22)
ggplot(mpg, aes(x = cty, y = hwy)) + geom_label(color = "blue")
ggplot(mpg, aes(x = cty, y = hwy)) + geom_quantile(color = "blue")
ggplot(mpg, aes(x = cty, y = hwy)) + geom_rug(sides = "bl")
ggplot(mpg, aes(x = cty, y = hwy)) + geom_smooth(method = "lm")

# discrete x, continuous y
ggplot(mpg, aes(x=class)) + geom_bar()
# col은 원소가 하나인 bar와 다르게 두개가 있어야 한다
ggplot(mpg, aes(x=class, y=hwy, fill=class)) + geom_col()
# boxplot은 상자 형태로 그린다
ggplot(mpg, aes(x=class, y=hwy, fill=class)) + geom_boxplot()
# dotplot은 동그란 점으로 그린다다
ggplot(mpg, aes(x=class, y=hwy, fill=drv)) + 
  geom_dotplot(binaxis = "y",stackdir = "center")
# violin
ggplot(mpg, aes(x=class, y=hwy, fill=drv)) + 
  geom_violin()
ggplot(mpg, aes(x=class, y=hwy, fill=class)) + 
  geom_violin(scale = "area")

mpg[ , c("class", "hwy")] %>% data.frame()
# ONE VARIABLE continuous
c <- ggplot(mpg, aes(hwy))
# density
c + geom_density()
# area
c + geom_area(stat = "bin")
# dotplot
c + geom_dotplot(binwidth =  2)+
  scale_fill_gradient(low = "red",high = "yellow")
# freqpoly
c + geom_freqpoly()
# histogram
c + geom_histogram(binwidth = 2,color ="white")
c + geom_histogram(bins = 8,color ="white")
c + geom_histogram(bins = 8,color ="white", fill = "steelblue")

colors() # R내의 색상 정보

#range와 같은거
seq(1,10)
seq(1,10,2)
length(seq(10,48,2))

df <- data.frame(col1 = 1:20,
                 col2 = seq(10, 48, 2))
df
class(df) # "data.frame"
class(mpg) # "tbl_df"  "tbl"  "data.frame"

tibble # data.frame의 단점을 극복한것

# tibble 형태로 바꾸는 as_tibble()
df <- as_tibble(df)
df

# vetor, dataframe, list

ggplot(data = mpg, aes(x = displ, y = hwy, color = drv))+
  geom_point(size = 3)
# shape 추가 (모양)
ggplot(data = mpg, aes(x = displ, y = hwy, color = drv, shape = drv))+
  geom_point(size = 3)
# smooth 추가 (음영추가)
ggplot(data = mpg, aes(x = displ, y = hwy, color = drv, shape = drv))+
  geom_point(size = 3)+
  geom_smooth()
# method 추가 (선 교정)
ggplot(data = mpg, aes(x = displ, y = hwy, color = drv, shape = drv))+
  geom_point(size = 3)+
  geom_smooth(method = "lm") # linear model
# theme 추가
ggplot(data = mpg, aes(x = displ, y = hwy, color = drv, shape = drv))+
  geom_point(size = 3)+
  geom_smooth(method = "lm")+
  theme_minimal()

install.packages("ggthemes")
library(ggthemes)
# theme_economist() 증권사 배경추가
ggplot(data = mpg, aes(x = displ, y = hwy, color = drv, shape = drv))+
  geom_point(size = 3)+
  theme_economist()
# theme_wsj() Wall Street Journal 배경 추가
ggplot(data = mpg, aes(x = displ, y = hwy, color = drv, shape = drv))+
  geom_point(size = 3)+
  theme_wsj()
# theme_solarized() Solarized palette 배경 추가
ggplot(data = mpg, aes(x = displ, y = hwy, color = drv, shape = drv))+
  geom_point(size = 3)+
  theme_solarized()
# theme_hc() Highcharts 배경
ggplot(data = mpg, aes(x = displ, y = hwy, color = drv, shape = drv))+
  geom_point(size = 3)+
  theme_hc()+
  ggtitle("배기량과 고속도로 주행연비의 관계") # 제목 추가
# lab을 이용하여 제목,범례 수정
ggplot(data = mpg, aes(x = displ, y = hwy, color = drv, shape = drv))+
  geom_point(size = 3)+
  theme_hc()+
  labs(title = "배기량과 고속도로 주행연비의 관계", 
       x= "배기량", y= "연비") + theme(title = element_text(size = 30))

# facet:그래프 분할
# facet_grid
d <- ggplot(mpg, aes(x=displ, y=hwy, color= drv))+
  geom_point()

d + facet_grid(drv ~ .)
d + facet_grid(. ~ cyl)
d + facet_grid(drv ~ cyl)
d + facet_grid(class ~ fl)
d + facet_grid(class ~ fl)
d + facet_grid(cty ~ .)

# facet_wrap 행또는 열이 많아지면 wrap
d + facet_grid(~ class)
d + facet_wrap(~ class)
d + facet_wrap(~ drv)

d + facet_wrap(~ class, nrow = 2)
d + facet_wrap(~ class, ncol = 4)

# jitter 많이 겹쳐있는 정도를 보여주는 명령어
ggplot(mpg, aes(x=displ, y=hwy, color= drv))+
  geom_point(size=3, position = "jitter")
# geom_jitter로도 활용가능
ggplot(mpg, aes(x=displ, y=hwy, color= drv))+
  geom_point(size=3)+
  geom_jitter(width = 0.5, height = 0.5)

# line
p1 <- ggplot(mpg, aes(x=displ, y=hwy, color= drv))
p1 + geom_point(size = 2)
p1 + geom_point(size = 2) + geom_line() #산점도 위에 선긋기
p1 + geom_line() # 선긋기만 그리기

# bar 응용
ggplot(data=mpg, aes(x= displ)) + geom_bar()
ggplot(data=mpg, aes(x= displ, fill = drv)) + geom_bar() # 색 구별
ggplot(data=mpg, aes(x= displ, fill = as.factor(cty))) + 
  geom_bar(position = "dodge") # dodge로 분리 
ggplot(data=mpg, aes(x= displ, fill = drv)) + geom_bar()+
  facet_wrap( ~ drv) # facet_wrap 으로 분리

# histogram
ggplot(data=mpg, aes(x= displ)) + geom_histogram(bins = 60)
ggplot(data=mpg, aes(x= displ)) + geom_histogram(bins = 10, color = "white")+
  geom_freqpoly(bins = 10, color = "red", size = 3) # histogram 위에 선긋기

ggplot(data=mpg, aes(x= displ)) + geom_histogram(binwidth = 1, color = "white")


mpg %>% select(manufacturer,model,hwy) %>% 
  arrange(desc(hwy)) %>% head()

mpg %>% select(manufacturer,model,hwy) %>%
  arrange(hwy) %>% head()

mpg %>% select(manufacturer,model,hwy,drv) %>%
  filter(drv == "f") %>% 
  arrange(hwy) %>% head(20)

ggplot(data = mpg, aes(x=drv, y=hwy)) + geom_point()

ggplot(data = mpg, aes(x=drv, y=hwy, color = hwy)) + 
  geom_point()

p <- ggplot(economics, aes(date, unemploy / pop))+
  geom_line()

ggplot(economics, aes(date, unemploy))+
  geom_line()

library(plotly)
# plots 가 아닌 Viewer로 보여줌
ggplotly(p)
ggplotly(ggplot(mpg, aes(x=displ, y=hwy, color = drv))+ geom_point())

diamonds
# 1~10중 5번 뽑기
sample(1:10, 5)
# replace로 중첩
sample(1:10, 15, replace = T)

sample(1:53950, 5394)

diamonds[sample(1:53950, 5394), ]
dia <- diamonds[sample(1:nrow(diamonds), nrow(diamonds)*0.1), ]

# 연습

# diamonds에 대하여 산점도를 아무거나 한개 그려주세요

ggplot(diamonds, aes(x=carat,y=price,color=color))+geom_point()

ggplot(mpg, aes(x=displ,y=hwy,color=drv))+geom_point()
ggplot(mpg, aes(x=displ,y=hwy,color=drv,shape=drv))+geom_point()
ggplot(mpg, aes(x=displ,y=hwy,color=drv,shape=drv))+geom_point()+
  facet_grid(drv ~.)
ggplot(mpg, aes(x=displ, fill = drv)) + geom_bar()
ggplot(mpg, aes(x=displ)) + geom_histogram(fill='blue', binwidth = 0.2)



ggplot(mpg, aes(x =class,fill = class)) + geom_bar()+
  ggtitle("클래스별 차종")+
  theme(legend.position = "none")

score1 <- data.frame(id = c("a","b","c","d","e"),
                     math = c(30,45,89,90,100))
score2 <- data.frame(id = c("a","b","f","g","h"),
                     english = c(55,80,65,88,95))

inner_join(score1,score2)
left_join(score1,score2)
right_join(score1,score2)
full_join(score1,score2)

names(diamonds)
ggplot(dia,aes(cut,fill=clarity)) +geom_bar()
ggplot(dia, aes(cut, fill=clarity)) + geom_bar()
ggplot(dia,aes(x=cut,fill=clarity))+geom_bar(position = "dodge")
ggplot(dia,aes(x=carat,color=cut))+geom_freqpoly(bins = 60)

airquality
air <- airquality
air %>% filter(Month == 5 | Month == 8) %>% 
  ggplot(aes(x=Day,y=Wind,color =as.factor(Month)))+geom_line()

economics
class(economics)

economics %>%  ggplot(aes(x = date,y = unemploy)) + geom_line()
economics %>%  ggplot(aes(x = date,y = unemploy/pop)) + geom_line()
