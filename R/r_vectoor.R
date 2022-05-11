x1 <- 1:20
x2 <- rep(c("a", "b"), 10)
x3 <- sample(1:100, 20)

a <- cbind(x1,x2,x3)
df_3 <- data.frame(a)
write.csv(df_3,'df_3.csv')
write.table(df_3,'df_3.txt', sep = ",")
write.table(df_3,'df_3.txt', sep = "\t")

read.csv("df_3.csv", header = F)
