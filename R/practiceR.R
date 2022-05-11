library(foreign)
library(dplyr)
library(ggplot2)

# spss 파일 읽기
raw_welfare <- read.spss("Koweps_hpc10_2015_beta1.sav")
raw_welfare

# 복사본 만들기
raw <- as.data.frame(raw_welfare)
raw <- as_tibble(rwa)

# 필요한 열만 뽑기
mpg

mpg$city <- mpg$cty
mpg <- mpg[,1:11]
mpg <- mpg[,-12]
colnames(mpg)[8] <- "city"
mpg %>% rename(city=cty)

welfare <- raw %>% 
  select(gender = h10_g3, birth = h10_g4,
         marriage = h10_g10, religion = h10_g11,
         income = p1002_8aq1, job = h10_eco9,
         region = h10_reg7)

# income 의 최대값은?
max(welfare$income, na.rm = T)
str(welfare)
summary(welfare)    

# income 열의 0을 NA로 처리하기
welfare$income <- ifelse(welfare$income == 0,NA,welfare$income)

boxplot(welfare$income)
welfare %>% ggplot(aes(y = income)) + geom_boxplot()
welfare %>% select(gender, income) %>%
  ggplot(aes(x= as.factor(gender), income)) + geom_boxplot()
ggplot(welfare,aes(factor(gender),income, fill = factor(gender))) + geom_boxplot()

ggplot(welfare,aes(x=income))+geom_density()
ggplot(welfare,aes(x=income))+geom_freqpoly()

ggplot(welfare,aes(x=income, color=factor(gender)))+geom_density()

welfare$gender <- ifelse(welfare$gender==1,"male","female")
table(welfare$gender)

ggplot(welfare,aes(x=gender,fill=gender))+geom_bar()

# 남여별 평균임금을 구하세요
names(welfare)
welfare %>% group_by(gender) %>% 
  summarise(mean=mean(income,na.rm=T)) %>%
  ggplot(aes(gender,mean,fill=gender))+geom_col()

welfare %>% select(gender,income) %>% 
  ggplot(aes(income,color=gender))+geom_density()

# 연령별 임금 구하기
range(welfare$birth)
welfare$age=2015-welfare$birth
welfare$age
welfare %>% filter(age >= 20) %>% 
  group_by(age,gender) %>% 
  summarise(mean_income = mean(income,na.rm=T)) %>% 
  ggplot(aes(age,mean_income,color=gender))+geom_line()


# 나이에 따른 소득
welfare <- welfare %>% 
  mutate(age_gen = ifelse(age < 30,"young",
                          ifelse(age<= 50,"middle","old")))
welfare %>% group_by(age_gen) %>% 
  summarise(mean_income = mean(income,na.rm = T)) %>% 
  ggplot(aes(x = age_gen,y = mean_income,fill=age_gen))+
  geom_col()+scale_x_discrete(limits=c("young","middle","old"))

welfare %>% group_by(age_gen,gender) %>% 
  summarise(mean_income = mean(income,na.rm = T)) %>% 
  ggplot(aes(x = age_gen,y = mean_income,fill=gender))+
  geom_col(position = "dodge")+scale_x_discrete(limits=c("young","middle","old"))


# air
airquality %>% filter(Month >=6 & Month <= 8) %>% 
  ggplot(aes(x= Ozone,y= Temp,color = factor(Month)))+geom_point()

sample(1:10,20,replace = T)

# sample을 이용해 diamonds의 행을 0.1정도로 뽑기

diamonds

idx = sample(1:nrow(diamonds),nrow(diamonds)*0.1,replace = F)

diamonds[idx, ]


# 70%를 뽑아서 test라는 이름으로 데이터프레임을 만들어 봅시다

idx2 = sample(1:nrow(diamonds),nrow(diamonds)*0.7,replace = F)

train <- diamonds[idx2, ]
test <- diamonds[-idx2, ] # 나머지 30%

# dplyr 의 sample
sample_n(diamonds,5394)
sample_frac(diamonds, 0.1)

#그래프의 범례 제거하기
theme(legend.position = "none")

# tibble 형태로 바꾸기
air <- as_tibble(airquality)

ggplot(data = mpg, aes(x = displ))+ geom_histogram(bins=10,color = "white")

# mpg에서 drv별로 hwy길을 boxplot으로 그려주세요
ggplot(mpg, aes(x=drv, y=hwy,fill= drv)) + geom_boxplot()
