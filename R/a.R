library(ggplot2)
ggplot(diamonds, aes(x = carat, y = price, color = color)) +
  geom_point()
