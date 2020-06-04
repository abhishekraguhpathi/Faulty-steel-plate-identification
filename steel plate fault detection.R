library(ggplot2)
library(caret)
library(tidyverse)
library(gridExtra)
library(e1071)
library(ROSE)

setwd("D:/Abhishek/steel plate fault detection")
getwd()
#Reading data
rm(steel_data_train)
steel_data_train<-read.csv(file.choose(),header = T)
test_data<-read.csv(file.choose(),header = T)
summary(steel_data_train)
str(steel_data_train)
str(steel_data_train$Faults)
steel_data_train$Faults<-factor(steel_data_train$Faults)
dim(steel_data_train)
#checking for NA values
any(is.na(steel_data_train))#no na values

#train and test data
set.seed(1234)
index<-createDataPartition(steel_data_train$Faults,p=0.7,list=F)
train_data<-steel_data_train[index,]
test_data<-steel_data_train[-index,]
ctrl<-trainControl(method = "repeatedcv",number = 10,repeats = 3)

#KNN
#model 1
model<-train(Faults~., 
             data=train_data,
             method="knn",
             metric="Accuracy", 
             trControl=ctrl, 
             tuneGrid=expand.grid(k=1:39 ))
plot(model)
pred<-predict(model,test_data)
confusionMatrix(pred,test_data$Faults)
table(train_data$Faults)

#since specificity and sensitivity  have a huge difference, it means that there is class imbalance
table(steel_data_train$Faults)
#to tackle class imbalance we will try oversampling and undersampling to see which is better
#oversampling
train_data_over<-ovun.sample(Faults~., data = train_data,method="over",N=1506)$data
table(train_data_over$Faults)

model2<-train(Faults~., 
             data=train_data_over,
             method="knn",
             metric="Accuracy", 
             trControl=ctrl, 
             tuneGrid=expand.grid(k=1:39 ))
varImp(model2)
pred2<-predict(model2,test_data)
confusionMatrix(pred2,test_data$Faults)

#feature selection
feature<-("Y_Perimeter,Square_Index ,TypeOfSteel_A400 ,TypeOfSteel_A300 ,X_Minimum,        
           Log_Y_Index ,Edges_Y_Index ,X_Perimeter,SigmoidOfAreas,Sum_of_Luminosity,
           X_Maximum,LogOfAreas,Pixels_Areas,Edges_X_Index,Empty_Index,Edges_Index,         
           Length_of_Conveyer,Y_Minimum ,Y_Maximum,Outside_X_Index ")

col<-c(7,17,12,13,1,24,20,6,27,8,2,22,5,19,16,15,11,4,3,18)
train_data_over_f<-train_data_over[,c(col)]
train_data_over_f$Faults<-train_data_over$Faults
dim(train_data_over_f)

#model 3 after feature selection and oversampling 
model3<-train(Faults~.,
              data=train_data_over_f,
              method="knn",
              metric="Accuracy",
              trControl=ctrl,
              tuneGrid=expand.grid(k=1:39 ))

plot(model3)
pred3<-predict(model3,test_data)  
confusionMatrix(pred3,test_data$Faults,positive = "1")  

table(train_data_over_f$Faults)

#undersampling as sensitivity and specificity are very far apart

train_data_under<-ovun.sample(Faults~.,data=train_data,method = "under",N=398)$data

train_data_under_f<-train_data_under[,col]  
train_data_under_f$Faults<-train_data_under$Faults  

#model4 after undersampling and feature selection
Knn_model4<-train(Faults~.,
              data=train_data_under_f,
              method="knn",
              metric="Accuracy",
              trControl=ctrl,
              tuneGrid=expand.grid(k=1:39))
  
  
plot(model4)
pred4<-predict(model4,test_data)
confusionMatrix(pred4,test_data$Faults)
varImp(Knn_model4)

#sensitivity and specificity a lot closer after under sampling
#but no information rate is more than accuracy , meaning stated accuracy of 56% is not true

#trying decision trees

install.packages("rpart")
library(rpart)
dtmodel1<-train(Faults~.,
                data=train_data_under_f,
                method="rpart",
                metric="Accuracy",
                trControl=ctrl,
                tuneLength=10)

dtpred1<-predict(dtmodel1,test_data)
confusionMatrix(dtpred1,test_data$Faults)


#Random forest
install.packages("randomForest")
library(randomForest)

rfmodel1<-train(Faults~.,
                data=train_data_under_f,
                method="rf",
                metric="Accuracy",
                trControl=ctrl,
                tuneLength=15)
plot(rfmodel1)
rfpred1<-predict(rfmodel1,test_data)     
confusionMatrix(rfpred1,test_data$Faults)
#accuracy more than no information rate,kappa>0.5,sensitivity and specificity are closer.
#accuracy=81.128%

rfpred1<-predict(rfmodel2,test_data)     
confusionMatrix(rfpred2,test_data$Faults)


#rfmodel=highest accuracy 
#rf final model
Actual_test_data<-read.csv(file.choose(),header=T)
table(steel_data_train$Faults)
steel_data_train_under<-ovun.sample(Faults~.,data=steel_data_train ,method = "under",N=566)$data

rf_final_model<-train(Faults~.,
                data=steel_data_train_under,
                method="rf",
                metric="Accuracy",
                trControl=ctrl,
                tuneLength=15
                )
rf_final_pred<-predict(rf_final_model,Actual_test_data)
write.csv(rf_final_pred, file="rf_final_prediction.csv")



