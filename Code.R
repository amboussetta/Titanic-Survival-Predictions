library(ggplot2)
library(dplyr)
library(tidyverse)
library(caret)
library(randomForest)
library(stringr)
library(party)
library(gridExtra)
library(knitr)
library(mice)


#________________________________________________#
############                          ############
############    Preparing the data    ############
############                          ############
#________________________________________________#

# read the csv files into our variables and create the full dataset
fulldataset <- read.csv("./train.csv")
set.seed(1)
test_index <- createDataPartition(y = fulldataset$Survived, times = 1, p = 0.2, list = FALSE)
trainset <- fulldataset[-test_index,]
testset <- fulldataset[test_index,]


# See how the daha looks:
head(fulldataset)

# Check for NAs:
colSums(is.na(trainset))

# Check for outliers:
boxplot(trainset[,c(6,7,8,10)])
summary(trainset[c(6,7,8,10)])

# Transform some variables to factors
vars <- c('PassengerId','Pclass','Sex','Embarked', 'Survived')
fulldataset[vars] <- lapply(fulldataset[vars], function(x) as.factor(x))


# Add new columns to the data set:
fulldataset <- mutate(fulldataset, Title = gsub('(.*, )|(\\..*)', '', fulldataset$Name), 
                      Agestage = ifelse(Age<18, "Child", "Adult"), 
                      Familysize = SibSp + Parch + 1)


# See the relation between sex, title
table(fulldataset$Sex, fulldataset$Title)
table(fulldataset$Agestage, fulldataset$Title)

# Titles with very low cell counts that should be grouped
rare <- c('Capt', 'Col', 'Don', 
          'Dona','Dr', 'Jonkheer', 'Lady', 'Major','Rev', 'Sir','the Countess')

# Also reassign mlle, ms, and mme accordingly
fulldataset$Title[fulldataset$Title == 'Mlle'] <- 'Miss' 
fulldataset$Title[fulldataset$Title == 'Ms'] <- 'Miss'
fulldataset$Title[fulldataset$Title == 'Mme'] <- 'Mrs' 
fulldataset$Title[fulldataset$Title %in% rare] <- 'Rare Title'

# Show title counts again
table(fulldataset$Title)


#Fill the age NAs with Mice:
mice <- mice(fulldataset[, !names(fulldataset) %in% c('PassengerId','Name',
                                                      'Ticket','Cabin','Familisize',
                                                      'Survived')], method='rf') 
mice_output <- complete(mice)
fulldataset$Age <- mice_output$Age

# Plot the two age distributions
par(mfrow=c(1,2))
hist(fulldataset$Age, freq=F, main='Age: Original Data', 
     col='darkblue', ylim=c(0,0.04))
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
     col='lightgreen', ylim=c(0,0.04))

# Add the variable mother
fulldataset$Mother <- 'Not Mother'
fulldataset$Mother[fulldataset$Sex == 'female' & fulldataset$Parch > 0 & 
                     fulldataset$Age > 18 & fulldataset$Title != 'Miss'] <- 'Mother'

# Splitting back the dataset into train and test
trainset <- fulldataset[-test_index,]
testset <- fulldataset[test_index,]

#________________________________________________#
############                          ############
############    Exploring the data    ############
############                          ############
#________________________________________________#

# Relation between Sex, Age and Survival
ggplot(trainset, aes(Age, fill = Survived)) + 
  geom_histogram() + 
  facet_grid(.~Sex)

# Relation between Survival, Fare, Family size and PcClass
trainset %>% 
  ggplot(aes(x = Fare, y = Survived)) +
  geom_jitter(aes_string(color = as.factor(trainset$Familysize)),
              size = as.factor(trainset$Familysize)) + 
  facet_grid(Sex~Pclass, scales = "free") + 
  scale_color_discrete("FamilySize") + 
  scale_size_discrete("FamilySize")

# Let's run ANOVA analysis to check for correlations
df <- trainset %>% select(Survived,Age,Fare, Pclass, Familysize)
df <- as.data.frame(lapply(df, function(x) as.numeric(x)))
fit <- aov(Survived ~ Age + Fare + Familysize + Pclass, data = df)
layout(matrix(c(1,2,3,4),2,2)) # optional layout 
plot(fit) 
summary(fit)

#________________________________________________#
############                          ############
############    Building the models   ############
############                          ############
#________________________________________________#

###### RANDOM FOREST #######

# Training the random forest model
trainset <- trainset %>% mutate_if(is.character, as.factor)
testset <- testset %>% mutate_if(is.character, as.factor)
rf_model <- randomForest(Survived ~ Sex + Age + SibSp + Parch + 
                           Pclass + Fare + Mother + Embarked + 
                           Title + Familysize, data = trainset)

# Show model error
plot(rf_model, ylim=c(0,0.4))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

# get the variable importance
rf_importance<- importance(rf_model)
var_importance <- data.frame(Variables = row.names(rf_importance), 
                            Importance = round(rf_importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rank_importance <- var_importance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rank_importance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            colour = 'red') +
  labs(x = 'Variables') 

# Make the predictions 

prediction <- predict(rf_model, testset)
RF_acc <- confusionMatrix(prediction,testset$Survived)
RF_acc

prediction <- as.numeric(prediction)
testsetpred<- as.numeric(testset$Survived)
RF_rmse <- RMSE(prediction,testsetpred)
RF_rmse

###### K-nearest neighbors #######

trainset <- fulldataset[-test_index,]
testset <- fulldataset[test_index,]

# Controlling the parameters of the training phase
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(1)

#train the model
knn_fit <- train(Survived ~ Sex + Age + Familysize + Fare + Title + Mother,
                 data = trainset, method = "knn",
                 trControl=trctrl,
                 preProcess = c("center", "scale"),
                 tuneLength = 10)

# Show the sumary
knn_fit

# Make predictions and get accuracy results
results<- predict(knn_fit, newdata=testset)
Knn_acc<- confusionMatrix(results, testset$Survived)
Knn_acc

prediction <- as.numeric(results)
testsetpred<- as.numeric(testset$Survived)
Knn_rmse <- RMSE(prediction,testsetpred)
Knn_rmse

#________________________________________________#
############                          ############
############    Printing the results  ############
############                          ############
#________________________________________________#

res <- data.frame(RandomForest = RF_acc$overall, Knn = Knn_acc$overall)
res["rmse", ] <- matrix(c(RF_rmse,Knn_rmse), ncol = 2)
print(format(res, scientific = FALSE), digits = 5)
