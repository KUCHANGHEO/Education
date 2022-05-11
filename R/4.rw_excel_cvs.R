getwd() # 현재 작업폴더 확인
setwd("C:/Rdata") # 새로운 작업폴더 지정

dir() # 폴더 내 파일 이름 보기

# readxl 패키지 설치, 로드
install.packages("readxl") 
library(readxl)

df_ex <- read_excel("excel_exam.xlsx")
mean(df_ex$math)

# txt
df_ex3 <- read.table("csv_exam.txt",header = T,sep = "\t")
write.table(df_ex3,"df_3")

# csv (comma separated values)
df_ex2 <- read.csv("csv_exam.csv")
df_ex2 <- df_ex2[1:10,]  
write.csv(df_ex2,"csv_exam_test.csv")
dir()


# 객체 그대로 보내고 싶을떄
x1 <- c(100, 80, 60, 40, 20)
x2 <- c("A", "B", "C", "A", "B")
df <- data.frame(score=x1,grade=x2)

save(df, file = 'df_midterm.rda')

rm(df)
load("df_midterm.rda")

# 읽기 : read.csv read_excel read.table
# 저장 : write.csv(.csv), save(.rda), write.table(.txt)

# excel_exam.xlsx 파일을 df_exam 이름으로 읽어 봅시다.

df_exam = read_excel("excel_exam.xlsx")
head(df_exam) # 앞에서 부터 일부행만 출력
head(df_exam, 2) # 2행만 출력
tail(df_exam) # 뒤에서 부터 출력
View(df_exam) # 데이터 뷰어창 열기
dim(df_exam) # 행, 열 출력
str(df_exam) # 데이터 속성 확인
summary(df_exam) # 요약 통계량

nrow(df_exam) # 행의 갯수 출력
ncol(df_exam) # 열의 갯수 출력
