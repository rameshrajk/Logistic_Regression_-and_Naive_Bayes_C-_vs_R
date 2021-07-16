#title: "CS 4375 Project 1 Naive Bayes"
#author: "Ramesh Kanakala"
#subtitle: "This is an R script with the purpose of running naive bayes on a 
#titanic data set to observe run time and other metrics"

### Logistic Regression
#load the data
ttnc <- read.csv(file = 'titanic_project.csv')
ttnc$pclass <- as.factor(ttnc$pclass)
ttnc$sex <- as.factor(ttnc$sex)
ttnc$survived <- as.factor(ttnc$survived)

#dividing into train/test, putting 75% in train
i <- 1:900
train <- ttnc[i,]
test <- ttnc[-i,]

start <- Sys.time() 
#train naive bayes model
library(e1071)
nb1 <- naiveBayes(as.factor(survived)~pclass+sex+age,family = "binomial", data = train)
end <- Sys.time()

#print probabilities from model
nb1

#test on test data
pred <- predict(nb1, newdata=test, type="class")

#print accuracy, sensitivity, and specificity #check spelling
library(caret) 
confusionMatrix(as.factor(pred), as.factor(test$survived))$overall[1]
confusionMatrix(as.factor(pred), as.factor(test$survived))$byClass[1]
confusionMatrix(as.factor(pred), as.factor(test$survived))$byClass[2]

confusionMatrix(as.factor(pred), as.factor(test$survived))
#time difference
end - start

require(dplyr)
require(ggplot2)

#DATA EXPLORATION: GRAPHS 1-4, FUNCTION 4-6
#graph exploration 1
ggplot(data = ttnc, mapping = aes(x = as.factor(survived), fill = as.factor(sex))) + 
  geom_bar() + 
  labs(c("0", "1"), title = "Titanic Survival by Sex") + scale_x_discrete(name = "Survived", labels=c("Perished", "Survived")) +
  scale_fill_discrete(name = "Sex", labels = c("Female", "Male"))
#graph exploration 2
ggplot(data = ttnc, mapping = aes(x = as.factor(survived), fill = as.factor(sex))) + 
  geom_bar() + 
  facet_wrap(~ pclass) +
  labs(c("0", "1"), title = "Titanic Survival by Class and Sex") + scale_x_discrete(name = "Survived", labels=c("Perished", "Survived")) +
  scale_fill_discrete(name = "Sex", labels = c("Female", "Male"))
#graph exploration 3
ggplot(data = ttnc, mapping = aes(as.factor(survived) , age, fill = as.factor(sex))) + 
  geom_boxplot() + 
  facet_wrap(~ pclass) +
  labs(c("0", "1"), title = "Titanic Survival Age Distribution by Class and Sex") + scale_x_discrete(name = "Survived", labels=c("Perished", "Survived")) +
  scale_fill_discrete(name = "Sex", labels = c("Female", "Male"))
#graph exploration 4
women.sub <- subset(ttnc, sex == 0)
ggplot(women.sub, aes(x="cc", y=as.factor(sex), fill=as.factor(age))) +
  geom_bar(stat="identity", width=1) +
  labs(title = "Age Distribution in Women", lab = "Women") +
  guides(fill=guide_legend(title="Age")) +
  coord_polar("y", start=0) + theme(
    legend.title = element_text(size = 5),
    legend.text = element_text(size = 5),
    legend.key.size = unit(.5, "line")
  )
#data exploration 4
nb1
##data exploration 5 + 6
head(ttnc)
tail(ttnc)
dim(ttnc)
