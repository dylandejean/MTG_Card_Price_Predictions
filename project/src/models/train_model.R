
# Libraries ---------------------------------------------------------------

library(data.table)
library(caret)
library(glmnet)
library(plotmo)


# Load train and test -----------------------------------------------------

train <- fread("./project/volume/data/interim/train.csv")
test <- fread("./project/volume/data/interim/test.csv")

### remove duplicates
train <- train[!duplicated(train),]

### keep future_price as train_y
train_y <- train$future_price


# Drop id -----------------------------------------------------------------

drops <- c('id')
train <- train[, !drops, with = FALSE]
test <- test[, !drops, with = FALSE]


# fit a linear model ------------------------------------------------------

model0 <- lm(future_price~., data=train)
summary(model0)
pred <- predict(model0, newdata=test)

sample_sub <- fread("./project/volume/data/raw/sample_sub.csv")
sample_sub$future_price <- pred
fwrite(sample_sub, "./project/volume/data/processed/submit_lm.csv")

# Make a submission file with merge
train <- fread('./project/volume/data/interim/train.csv')
test <- fread('./project/volume/data/interim/test.csv')
train <- train[!duplicated(train),]
train_y <- train$future_price

drops <- c('id')
model0 <- lm(future_price ~ ., data=train[, !drops, with = F])
pred <- predict(model0, newdata=test[, !drops, with = F])
result <- data.table(id=test$id, future_price = pred)

sample_sub <- fread('./project/volume/data/raw/sample_sub.csv')
sample_sub$future_price <- NULL

setkey(sample_sub, id)
setkey(result, id)
sample_sub <- merge(sample_sub, result, sort = F)
fwrite(sample_sub, './project/volume/data/processed/submit_lm.csv')

# fit a lasso model -------------------------------------------------------

drops <- c('id')
train <- train[, !drops, with=F]
test <- test[, !drops, with=F]

# Make dummy variables ----------------------------------------------------

### make fake future_price for dummyVars
test$future_price <- 0

### work with dummies
dummies <- dummyVars(future_price ~ ., data=train)
train <- predict(dummies, newdata=train)
test <- predict(dummies, newdata=test)

### convert them back to data.table
train <- data.table(train)
test <- data.table(test)


# Fit a model using cross-validation --------------------------------------

### glmnet needs matrix
train <- as.matrix(train)

### fit a model
model1 <- cv.glmnet(train, train_y, alpha=1, family="gaussian")

# alpha = 1 means lasso model
# alpha = 0 means ridge model
# alpha = 0.5 means mixture

### choose the best lambda
best_lambda <- model1$lambda.min

### see what predictors chosen from the lambda
predict(model1, s=best_lambda, newx=test, type="coefficients")


# Fit a full model --------------------------------------------------------

model2 <- glmnet(train, train_y, alpha=1, family="gaussian")

plot_glmnet(model2)

### save the model
saveRDS(model2,"./project/volume/models/model.model")


# Predict by the model ----------------------------------------------------

### glmnet needs matrix
test <- as.matrix(test)
pred <- predict(model1, s=best_lambda, newx=test)

# Make a model testing different alphas
model3 <- cv.glmnet(train, train_y, alpha = 0.7, family = "gaussian")
best_lambda3 <- model3$lambda.min
test <- as.matrix(test)
pred <- predict(model3, s=best_lambda3, newx=test)

# Remove potential unnecessary predictors
train <- data.table(train)
trainRarityOnly <- train[, .(rarityCommon, rarityMythic, rarityRare, rarityUncommon, legendary)]
trainRarityOnly <- as.matrix(trainRarityOnly)
test <- data.table(test)
testRarityOnly <- test[, .(rarityCommon, rarityMythic, rarityRare, rarityUncommon, legendary)]
testRarityOnly <- as.matrix(testRarityOnly)
model4 <- cv.glmnet(trainRarityOnly, train_y, alpha=1, family ="gaussian" )
best_lambda4 <- model4$lambda.min
test <- as.matrix(test)
pred <- predict(model4, s=best_lambda4, newx=testRarityOnly)

# Try another removal
train <- data.table(train)
trainRarityOnly <- train[, .(rarityCommon, rarityMythic, rarityRare, rarityUncommon)]
trainRarityOnly <- as.matrix(trainRarityOnly)
test <- data.table(test)
testRarityOnly <- test[, .(rarityCommon, rarityMythic, rarityRare, rarityUncommon)]
testRarityOnly <- as.matrix(testRarityOnly)
model5 <- cv.glmnet(trainRarityOnly, train_y, alpha=1, family ="gaussian" )
best_lambda5 <- model5$lambda.min
test <- as.matrix(test)
pred <- predict(model5, s=best_lambda5, newx=testRarityOnly)

# Try ridge regression with further reduced data
model6 <- cv.glmnet(trainRarityOnly, train_y, alpha=0, family ="gaussian" )
best_lambda6 <- model6$lambda.min
test <- as.matrix(test)
pred <- predict(model6, s=best_lambda6, newx=testRarityOnly)

saveRDS(model6,"./project/volume/models/ridge_model.model")


# Make a submission file --------------------------------------------------

sample_sub <- fread("./project/volume/data/raw/sample_sub.csv")

sample_sub$future_price <- pred

fwrite(sample_sub, "./project/volume/data/processed/submit_ridge.csv")
