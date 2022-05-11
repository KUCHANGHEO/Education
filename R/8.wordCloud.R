install.packages("multilinguer")
multilinguer::install_jdk()

install.packages("KoNLP", 
                 repos = c("https://forkonlp.r-universe.dev",
                           "https://cloud.r-project.org"),
                 INSTALL_opts = c("--no-multiarch")
)
library(KoNLP)
library(wordcloud2)
library(ggplot2)
library(dplyr)
useSejongDic()
getwd()

# 파일 읽기
text = readLines("jeonduhwan.txt",encoding = "UTF-8")
# 파일속 단어 나누기
nouns <-  extractNoun(text)

typeof(nouns)
# unlist화 시킨후 한자릿수 단어 지우기
nouns <- unlist(nouns)
nouns <- nouns[nchar(nouns) >= 2]
# table로 저장하고 sort로 정렬하기
wordFreq <- table(nouns)
wordFreq <- head(sort(wordFreq,decreasing = T), 25)
# wordCloud로 표현
wordcloud2(wordFreq)

# 간단히

nouns <- extractNoun(text) %>% unlist()
nouns <- nouns[nchar(nouns) >= 2]
words <- table(nouns) %>% sort(decreasing = T) %>% head(25)
wordcloud2(words, fontFamily = "나눔고딕")

text <- readLines("kimdaejung.txt",encoding = "UTF-8")
nouns <- extractNoun(text) %>% unlist()
nouns <- nouns[nchar(nouns) >1] %>% 
  table() %>% sort(decreasing = T) %>% 
  head(20) %>% wordcloud2()

words <- data.frame(nouns)
names(words)[1] <- "word"
words %>% arrange(desc(Freq)) %>% 
  filter(Freq > 5)

ggplot(data = words, aes(x= reorder(word,-Freq), y = Freq))+
  geom_col() + theme(legend.position = "None")

ggplot(data = words, aes(x= reorder(word,Freq), y = Freq))+
  geom_col() + theme(legend.position = "None")+
  coord_flip()


# 김대중 대통령 취임사의
# 빈도수 상위 10개 어휘를 워드클라우드로
# 만들고,coord_filp()인 막대 그래프를 그리자자

text <- readLines("kimdaejung.txt",encoding = "UTF-8")
nouns <- extractNoun(text) %>% unlist()
nouns <- nouns[nchar(nouns) >1] %>% 
  table() %>% sort(decreasing = T) %>% head(20)

words <- data.frame(nouns)
names(words)[1] <- "word"
words %>% arrange(desc(Freq)) %>% 
  filter(Freq > 5)
ggplot(data = words, aes(x= reorder(word,Freq), y = Freq,fill=word))+
  geom_col() + theme(legend.position = "None")+
  coord_flip()
