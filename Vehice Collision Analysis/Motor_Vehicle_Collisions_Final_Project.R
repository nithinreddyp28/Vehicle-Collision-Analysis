# Preset
# Preset
cat("\014") # clears console
rm(list = ls()) # clears global environment
try(dev.off(dev.list()["RStudioGD"]), silent = TRUE) # clears plots
try(p_unload(p_loaded(), character.only = TRUE), silent = TRUE) #clears packages
options(scipen = 100) # disables scientific notion for entire R session

# Used libraries
library(ggplot2)
library(pacman)
library(tidyverse)
library(tidyr)
library(janitor)
library(dplyr)
#install.packages("Hmisc")
library(Hmisc)
#install.packages("corrplot")
library(corrplot)
#install.packages('car')
library(car)
#install.packages("leaps")
library(leaps)
#install.packages("ISLR")
library(ISLR)
#install.packages("glmnet")
library(glmnet)
#install.packages('caret')
library(caret)
#install.packages("Metrics")
library(pROC)
library(Metrics)


####ALY 6015- Final Project: Initial Analysis Report####

################################

ds <- read.csv('motor_vehicle_collisions.csv')
#summary(collisions)
#describe(collisions)
ds <- clean_names(ds)
str(ds)
#describe(ds)
ds <- subset(ds, select=-c(on_street_name,cross_street_name,off_street_name,
                           contributing_factor_vehicle_3, contributing_factor_vehicle_4, 
                           contributing_factor_vehicle_5,vehicle_type_code_3,
                           vehicle_type_code_4,vehicle_type_code_5))

# Removing rows where with empty cells
# Replace empty strings with NA
ds[ds == ""] <- NA

# Remove rows with NAs in specific columns
ds = ds[complete.cases(ds$latitude, ds$longitude,ds$zip_code,
                       ds$contributing_factor_vehicle_1,ds$contributing_factor_vehicle_2,
                       ds$vehicle_type_code_1,ds$vehicle_type_code_2), ]

##Handling Data and year

ds$crash_date <- as.Date(ds$crash_date, format = "%m/%d/%Y")
ds$Year <- as.integer(format(ds$crash_date, "%Y"))

ds <- subset(ds, Year %in% c(2021,2022, 2023))

#creating a new column for fatalities

ds$fatalities <- ifelse(ds$number_of_persons_injured > 0, 1, 0)

###EDA####

##correlation plot to find the relation between variables

##Creating subset to build correlation matrix
subset_for_cor <- subset(ds, select = c(-borough, -contributing_factor_vehicle_1, 
                                        -vehicle_type_code_1, -crash_date,-crash_time,-zip_code,-Year,
                                        -location,-contributing_factor_vehicle_2,-vehicle_type_code_2,
                                        -latitude,-longitude))

# Remove non-numeric columns
subset_for_cor<- subset_for_cor[, sapply(subset_for_cor, is.numeric)]

# Creating a correlation matrix
cor_matrix <- cor(subset_for_cor)

# Opening a new graphics window
dev.new()

# Plotting the correlation matrix
corrplot(cor_matrix, method = "number")

# Crash Number Distribution
ggplot(ds,aes(x=Year))+
  geom_bar()+
  labs(title = "Crash Number Distribution from 2021-2023",x="Year",y="Count")+
  theme_classic()

##Grouping by Year
df_group <- ds %>% group_by(Year) %>% 
  summarise(number_of_persons_injured = sum(number_of_persons_injured),
            number_of_persons_killed = sum(number_of_persons_killed),
            number_of_pedestrians_injured = sum(number_of_pedestrians_killed),
            number_of_pedestrians_killed = sum(number_of_pedestrians_killed),
            number_of_motorist_injured = sum(number_of_motorist_injured),
            number_of_motorist_killed = sum(number_of_motorist_killed),
            number_of_cyclist_injured = sum(number_of_cyclist_injured),
            number_of_cyclist_killed = sum(number_of_cyclist_killed))

#Deaths by year

df_killed <- ds %>% group_by(Year) %>% 
  summarise(number_of_persons_killed = sum(number_of_persons_killed),
            number_of_pedestrians_killed = sum(number_of_pedestrians_killed),
            number_of_motorist_killed = sum(number_of_motorist_killed),
            number_of_cyclist_killed = sum(number_of_cyclist_killed))

# Number of people killed (Change-1:lets change chart)
df_long <- gather(df_killed, key = "Variable", value = "Value", -Year,-number_of_persons_killed)

#ggplot(df_long, aes(x = Year, y = Value, fill = Variable)) +
  #geom_bar(stat = "identity", position = "stack") +
  #labs(title = "Distribution of Number of People Killed from 2021-2023",
       #y = "Total Value", fill = "Number of Persons Killed")

ggplot(df_long, aes(x = Year, y = Value, color = Variable)) +
  geom_line() +
  geom_text(aes(label = Value), vjust = -0.5, hjust = 0.5, size = 3) +  # Add labels
  labs(title = "Distribution of Number of People Killed from 2021-2023",
       y = "Total Value", color = "Number of Persons Killed") +
  scale_color_manual(values = c("#FF5733", "#33FF57", "#3357FF"), name = "Variables") +
  guides(color = guide_legend(title = "Variables"))

# Contributing Factors for crashes

df_factors <- table(ds$contributing_factor_vehicle_1)
df_factors <- as.data.frame(df_factors)

new_column_names <- c("Factors_for_accident", "Frequency")
names(df_factors) <- new_column_names
colnames(df_factors)
df_factors <- df_factors %>% arrange(desc(Frequency)) %>% slice(1:10)

ggplot(df_factors, aes(x = Factors_for_accident, y = Frequency)) +
  geom_bar(stat = "identity", position = "stack") +
  labs(title = "Top 10 factors for accident and count ",
       y = "Total Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

##Fatalities and Boroughs
# Frequency table
fatalities_table <- table(ds$borough, ds$fatalities)
print(fatalities_table)

# Count plot
ggplot(ds, aes(x = borough, fill = factor(fatalities))) +
  geom_bar(position = "stack") +
  labs(title = "Distribution of Fatalities by Boroughs", x = "Borough", y = "Count") +
  scale_fill_manual(values = c("0" = "blue", "1" = "red")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

##Multiple Linear Regression

##building a subset model to find best model for regression

subsets_model <- regsubsets(number_of_persons_injured ~
                              number_of_persons_killed +number_of_pedestrians_injured+
                              number_of_pedestrians_killed + number_of_cyclist_injured +
                              number_of_cyclist_killed + number_of_motorist_injured +
                              number_of_motorist_killed, data = subset_for_cor, nbest=3)
summary(subsets_model)

#Regression Analysis

model <- lm(number_of_persons_injured ~ number_of_pedestrians_injured + number_of_cyclist_injured 
            + number_of_motorist_injured + number_of_persons_killed, 
            data = subset_for_cor)
summary(model)

#Rgeression Diagnostics 

plot(model)

#Evaluating multi-collinearity

vif(model) 
sqrt(vif(model)) > 2

# Assessing outliers

outlierTest(model=model)
#Treating outliers

subset_for_cor <- subset_for_cor[!(1:nrow(subset_for_cor) %in% c(129399,130829,1986402,2008327,1931080,1963551,
                                                                 1999539,2006460,1610)), ]

model2 <- lm(number_of_persons_injured ~ number_of_pedestrians_injured + number_of_cyclist_injured 
             + number_of_motorist_injured + number_of_persons_killed, 
             data = subset_for_cor)

summary(model2)

AIC(model,model2)

##both models are the same

##ANOVA## Change:2; should mention in the document that we made changes 

#H0: There is no significant difference in the mean number of persons killed among different levels of contributing_factor_vehicle

#H1: At least one group's mean severity is different from the others.

model_anova <- aov(number_of_persons_killed ~ contributing_factor_vehicle_1 + 
                     contributing_factor_vehicle_2, data = ds)

summary(model_anova)

##Reject H0 as p is less than 0.05 for contributing_factor_vehicle_1 but not for contributing_factor_vehicle_2.

###Chi Square Test###

#H0: There is no association between the borough and the occurrence of fatalities in traffic collisions.

#H1: There is an association between the borough and the occurrence of fatalities in traffic collisions.

# Buinding a table for borough and fatalities
table_borough_fatalities <- table(ds$borough, ds$fatalities)

# Chi-square test for borough and fatalities
chi_sq_test_borough <- chisq.test(table_borough_fatalities)

chi_sq_test_borough

##We do reject H0; you would reject the null hypothesis. This suggests that there is strong evidence to conclude that there is an association between the borough and the occurrence of fatalities in traffic collisions


##########GLM Starts#####

#GLM Logistic
set.seed(3456)

trainIndex <- createDataPartition(ds$number_of_persons_injured, p = 0.70, list = FALSE, times = 1)
caret_train <- ds[ trainIndex,]
caret_test <- ds[-trainIndex,]

caret_train$contributing_factor_vehicle_2 <- as.factor(caret_train$contributing_factor_vehicle_2)
caret_test$contributing_factor_vehicle_2 <- as.factor(caret_test$contributing_factor_vehicle_2)


model1= glm(fatalities~ contributing_factor_vehicle_1+contributing_factor_vehicle_2
            ,data=caret_train,family=binomial(link="logit"))
summary(model1)
coef(model1)

caret_train$fatalities <- as.factor(caret_train$fatalities)

# Make predictions on the train set
predictions_train <- predict(model1, newdata = caret_train, type = "response")
predicted_classes_train <- ifelse(predictions_train >= 0.5, 1, 0)
predicted_classes_train <- factor(predicted_classes_train, levels = levels(caret_train$fatalities))

# Now, you can check the confusion matrix
confusionMatrix(predicted_classes_train, caret_train$fatalities, positive = "1")

caret_test$fatalities <- as.factor(caret_test$fatalities)

# Make predictions on the test set

new_levels <- setdiff(levels(caret_test$contributing_factor_vehicle_2), levels(caret_train$contributing_factor_vehicle_2))
print(new_levels)

caret_test <- caret_test[!(caret_test$contributing_factor_vehicle_2 %in% new_levels), ]
predictions <- predict(model1, newdata = caret_test, type = "response")
predicted_classes <- ifelse(predictions >= 0.5, 1, 0)
predicted_classes <- factor(predicted_classes, levels = levels(caret_test$fatalities))

# Now, you can check the confusion matrix
confusionMatrix(predicted_classes, caret_test$fatalities, positive = "1")

#analysis on Accuracy, Precision, Recall, Specificity
TP = 4168
FN = 2957
TN = 18417
FP = 8858

#calculating recall
recall = TP / (TP + FN)
recall

#calculating Precision
Precision = TP / (TP + FP)
Precision

#calculating F1 Score
F1 = 2 * (Precision * recall) / (Precision + recall)
F1

#Plotting ROC curve

roc1 <- roc(caret_test$fatalities,predictions)
plot(roc1,col="blue")

#interpreting AUC
auc_value <- roc1$auc
auc_value
## GLM ENDS##

###Ridge and Lasso###

set.seed(123)

dq <- subset(ds, select=c("contributing_factor_vehicle_1","contributing_factor_vehicle_2",
                          "vehicle_type_code_1","vehicle_type_code_2", "fatalities"))
trainIndex <- sample(x=nrow(dq), size=nrow(dq)*0.7)

train <- dq[trainIndex,]
test <- dq[-trainIndex,]

train_subset <- train[1:1000, ]
test_subset = test[1:1000, ]

##As glmnet requires a matrix, converting the data in to matrix
train_x <- model.matrix(fatalities ~ ., data = train_subset)[, -1]
test_x <- model.matrix(fatalities~., data = test_subset)[,-1]

##Assigning grade to train and test variable
train_y<-train_subset$fatalities
test_y<-test_subset$fatalities


##Finding best lambda values 
set.seed(123)
cv.lasso <- cv.glmnet(train_x, train_y, nfolds=10)
plot(cv.lasso)
cv.lasso$lambda.min
cv.lasso$lambda.1se

log(cv.lasso$lambda.min)
log(cv.lasso$lambda.1se)

##Fitting models based in Lambda##

#building model on training data using lambda.min

#alpha =1 <we specify that we are running a Lasso model>
#alpha=0 <we specify that we are running Ridge model>

##Lasso Regression##
lasso_train_model_min <- glmnet(train_x, train_y, alpha=1, lambda = cv.lasso$lambda.min)
lasso_train_model_min 

#display coefficients
coef(lasso_train_model_min )

#building model on training data using lambda.1se
lasso_train_model_1se <- glmnet(train_x, train_y, alpha=1, lambda = cv.lasso$lambda.1se)
lasso_train_model_1se 

#display coefficients
coef(lasso_train_model_1se)

# Predicting based on train data using lambda.1se
predit_train_lasso_1se <- predict(lasso_train_model_1se, newx = train_x)
rmse_train_lasso_1se <- rmse(train_y, predit_train_lasso_1se)
rmse_train_lasso_1se

# Predicting based on test data using lambda.1se
predit_test_lasso_1se <- predict(lasso_train_model_1se, newx = test_x)
rmse_test_lasso_1se <- rmse(test_y, predit_test_lasso_1se)
rmse_test_lasso_1se

## Predicting based on train data using lambda.min
predit_train <- predict(lasso_train_model_min ,newx=train_x)
rmse_train <- rmse(train_y,predit_train )
rmse_train 

## Predicting based on test data using lambda.min

predit_test <- predict(lasso_train_model_min ,newx=test_x)
rmse_test <- rmse(test_y,predit_test)
rmse_test

##Ridge Regression###

#building model on training data using lambda.1se

regid_train_model_1se <- glmnet(train_x, train_y, alpha=0, lambda = cv.lasso$lambda.1se)
regid_train_model_1se 

#display coefficients

coef(regid_train_model_1se)

#building model on training data using lambda.min

regid_train_model_min <- glmnet(train_x, train_y, alpha=0, lambda = cv.lasso$lambda.min)
regid_train_model_min 

#display coefficients

coef(regid_train_model_1se)

## Predicting based on train data using lambda.1se
predit_train2 <- predict(regid_train_model_1se ,newx=train_x)
rmse_train2 <- rmse(train_y,predit_train2)
rmse_train2
## Predicting based on test data using lambda.1se
predit_test2 <- predict(regid_train_model_1se ,newx=test_x)
rmse_test2 <- rmse(test_y,predit_test2)
rmse_test2
## Predicting based on train data using lambda.min
predit_train3 <- predict(regid_train_model_min ,newx=train_x)
rmse_train3 <- rmse(train_y,predit_train3)
rmse_train3
## Predicting based on test data using lambda.min
predit_test3 <- predict(regid_train_model_min ,newx=test_x)
rmse_test3 <- rmse(test_y,predit_test3)
rmse_test3
