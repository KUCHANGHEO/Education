
cars
plot(cars)

install.packages("ggplot2")
install.packages("dplyr")
library(dplyr)

library(ggplot2)

cars %>% ggplot(aes(x = speed, y=dist)) +
  geom_point(color = 'skyblue', size = 3)+
  geom_smooth(method = 'lm')

# 상관관계
cor(cars)

cor.test(cars$speed, cars$dist)

corrplot::corrplot.mixed(cor(cars))

##
mtcars
cor(mtcars)
corrplot::corrplot.mixed(cor(mtcars))


#
fit.cars = lm(dist ~ speed, data = cars)


summary(fit.cars)
str(fit.cars)
names(fit.cars)
attach(fit.cars)
coefficients

residuals


new <- data.frame(speed= c(122))
class(new)
predict(fit.cars, newdata = new)

new <- data.frame(speed= c(30))
predict(fit.cars, newdata = new)

new <- data.frame(speed= c(30, 31, 32, 33))
predict(fit.cars, newdata = new)

new <- data.frame(speed= c(30))
predict(fit.cars, newdata = new, interval = "confidence")

new <- data.frame(speed= c(30, 31, 32, 33))
predict(fit.cars, newdata = new, interval = "confidence")

predict(fit.cars, newdata = new, interval = "confidence", level = 0.9)

predict(fit.cars, newdata = new, interval = "prediction")

## fit.cars
plot(fit.cars, 1)
plot(fit.cars, 2)
plot(fit.cars, 1)
plot(fit.cars, 1)

install.packages("lmtest")
install.packages("car")
library(lmtest)
library(car)
bptest(fit.cars)
ncvTest(fit.cars)

shapiro.test(fit.cars$res)

9
##

data <- cars
data$dist <- sqrt(data$dist)
colnames(data)[2] <- "sqrt.dist"
head(data,5)

fit.cars2 <- lm(sqrt.dist~speed, data = data)

bptest(fit.cars2)

ncvTest(fit.cars2)

shapiro.test(fit.cars2$residuals)

summary(fit.cars2)

ggplot(data=data, aes(x=speed, y=sqrt.dist))+
  geom_point()+
  geom_abline(data=data,
              intercept = 1.27705, slope = 0.32241,
              col="red")

#########
mtcars
# hp에 따른 mpg의 변화

fit.mpg = lm(mpg ~ hp, data = mtcars)
summary(fit.mpg)



data <- read.csv("BostonHousing.csv")
head(data)

lm(formula = medv ~ ., data = data)

house <- data

cor(house)
corrplot::corrplot.mixed(cor(house))

psych::pairs.panels(house[names(data)])


# 다중 회귀분석

data_lm <- lm(medv ~ . , data= house)

data_lm2 <- lm(medv ~ .^2 , data= house)

summary(data_lm)

summary(data_lm2)

data_lm_full <- lm(medv ~ ., data = )


new <- house[1:3,1:13]
predict(data_lm2, newdata = new)


set.seed(1234)
n <- nrow(house)
idx <- 1:n
training_idx <- sample(idx, n * .70)
idx <- setdiff(idx, training_idx)
validate_idx <- sample(idx, n * .20)

test_idx <- setdiff(idx, validate_idx)
training <-  house[training_idx,]

test <-  house[-training_idx,]

trainging_lm <- lm(medv ~ . , data = training)

round(predict(trainging_lm, newdata = test[, 1:13]),1)
y_traing <- round(predict(trainging_lm, newdata = test[, 1:13]),1)
y_forward <- round(predict(data_forward, newdata = test[, 1:13]),1)
y_back <- round(predict(data_backward, newdata = test[, 1:13]),1)
y_both <- round(predict(data_both, newdata = test[, 1:13]),1)

test[,14]



# measuring the quality of fit
mse <- function(yi, yhat_i){
  (mean((yi-yhat_i)^2))
}

rmse <- function(yi, yhat_i){
  sqrt(mean((yi-yhat_i)^2))
}

mae <- function(yi, yhat_i){
  mean(abs(yi-yhat_i))
}

mape <- function(yi, yhat_i){
  mean(abs((yi-yhat_i)/yi))*100
}


myfcn_measures <- function(yi, yhat_i){
  c(mse(yi, yhat_i), rmse(yi, yhat_i), mae(yi, yhat_i), mape(yi, yhat_i))
}

data_lm_full2 <- lm(medv ~ .^2 , data = training)

data_lm_full2

length(data_lm_full2)

library(MASS)
data_forward <- stepAIC(data_lm_full, direction = "forward",
                        scope = list(upper = ~.^2, lower = ~1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          ),
                        trace=FALSE)

data_backward <- stepAIC(data_lm_full, direction = "backward",
                         scope = list(upper = ~.^2, lower = ~1),
                         trace=FALSE)
data_both<- stepAIC(data_lm_full, direction = "both",
                    scope = list(upper = ~.^2, lower = ~1),
                    trace=T)

length(coef(data_forward))
length(coef(data_backward))
length(coef(data_both))

round(predict(trainging_lm, newdata = test[ ,1:13]), 1)
round(predict(data_forward, newdata = test[ ,1:13]), 1)
round(predict(data_backward, newdata = test[ ,1:13]), 1)
round(predict(data_both, newdata = test[ ,1:13]), 1)


# 다중공선성
car::vif(trainging_lm)
car::vif(data_backward)
car::vif(data_both)

lm(medv ~ rm+dis+ptratio+lstat, data = training)

