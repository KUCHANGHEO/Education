getwd()
setwd("C:/Rdata")

dir()

install.packages("readxl")
library(readxl)
df_ex <- read_excel("excel_exam.xlsx")
mean(df_ex$math)

# csv (comma separated values)
df_ex2 <- read.csv("csv_exam.csv")
df_ex2 <- df_ex2[1:10,]  
write.csv(df_ex2,"csv_exam_test.csv")
dir()
