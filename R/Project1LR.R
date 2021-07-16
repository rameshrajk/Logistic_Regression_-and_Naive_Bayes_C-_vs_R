#title: "CS 4375 Project 1 Logistic Regression"
#author: "Ramesh Kanakala"
#subtitle: "This is an R script with the purpose of running logistic regression 
#on a titanic data set to observe run time and other metrics"

### Logistic Regression
#load the data
ttnc <- read.csv(file = 'titanic_project.csv')
#ttnc$pclass <- as.factor(ttnc$pclass)
ttnc$sex <- as.factor(ttnc$sex)
ttnc$survived <- as.factor(ttnc$survived)

#dividing into train/test, putting 75% in train
i <- 1:900
train <- ttnc[i,]
test <- ttnc[-i,]

start <- Sys.time() 
#train logistic regression model
glm1 <- glm(survived~pclass, family = "binomial", data = train)
end <- Sys.time()

#print coefficients of model
glm1$coefficients[]

#test on test data
probs <- predict(glm1, newdata=test, type="response")
pred <- ifelse(probs>0.5, 1, 0)

#print accuracy, sensitivity, and specificity #check spelling
library(caret) 
confusionMatrix(as.factor(pred), as.factor(test$survived))$overall[1]
confusionMatrix(as.factor(pred), as.factor(test$survived))$byClass[1]
confusionMatrix(as.factor(pred), as.factor(test$survived))$byClass[2]

#time difference
end - start

#DATA EXPLORATION: FUNCTIONS 1-3
#data exploration 1
str(ttnc)
#data exploration 2
summary(ttnc)
#data exploration 3
summary(glm1)

confusionMatrix(as.factor(pred), as.factor(test$survived))
