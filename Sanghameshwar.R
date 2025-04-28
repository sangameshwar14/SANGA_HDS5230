library(mlbench)
library(purrr)

data("PimaIndiansDiabetes2")
ds <- as.data.frame(na.omit(PimaIndiansDiabetes2))
## fit a logistic regression model to obtain a parametric equation
logmodel <- glm(diabetes ~ .,
                data = ds,
                family = "binomial")
summary(logmodel)

cfs <- coefficients(logmodel) ## extract the coefficients
prednames <- variable.names(ds)[-9] ## fetch the names of predictors in a vector
prednames

sz <- 100000000 ## to be used in sampling
##sample(ds$pregnant, size = sz, replace = T)

dfdata <- map_dfc(prednames,
                  function(nm){ ## function to create a sample-with-replacement for each pred.
                    eval(parse(text = paste0("sample(ds$",nm,
                                             ", size = sz, replace = T)")))
                  }) ## map the sample-generator on to the vector of predictors
## and combine them into a dataframe

names(dfdata) <- prednames
dfdata

class(cfs[2:length(cfs)])

length(cfs)
length(prednames)
## Next, compute the logit values
pvec <- map((1:8),
            function(pnum){
              cfs[pnum+1] * eval(parse(text = paste0("dfdata$",
                                                     prednames[pnum])))
            }) %>% ## create beta[i] * x[i]
  reduce(`+`) + ## sum(beta[i] * x[i])
  cfs[1] ## add the intercept

## exponentiate the logit to obtain probability values of thee outcome variable
dfdata$outcome <- ifelse(1/(1 + exp(-(pvec))) > 0.5,
                         1, 0)

library(xgboost)
library(dplyr)
library(caret)
sizes <- c(100, 1000, 10000, 100000, 1000000, 10000000)

results <- data.frame(Size = integer(), Accuracy = numeric(), Time = numeric())

set.seed(123)

for (sz in sizes) {
  # Sample from dfdata
  dssample <- dfdata %>%
    sample_n(size = sz, replace = FALSE)
  
  # Split into train/test
  trainIndex <- createDataPartition(dssample$outcome, p = 0.8, list = FALSE)
  trainData <- dssample[trainIndex, ]
  testData <- dssample[-trainIndex, ]
  
  # Prepare matrices
  train_matrix <- xgb.DMatrix(data = as.matrix(trainData %>% select(-outcome)),
                              label = trainData$outcome)
  test_matrix <- xgb.DMatrix(data = as.matrix(testData %>% select(-outcome)),
                             label = testData$outcome)
  
  # Train and time the model
  start_time <- Sys.time()
  model <- xgboost(data = train_matrix,
                   objective = "binary:logistic",
                   nrounds = 100,
                   verbose = 0)
  end_time <- Sys.time()
  
  # Predict and calculate accuracy
  preds <- predict(model, test_matrix)
  preds_class <- ifelse(preds > 0.5, 1, 0)
  acc <- mean(preds_class == testData$outcome)
  
  # Save results
  results <- rbind(results, data.frame(
    Size = sz,
    Accuracy = acc,
    Time = as.numeric(difftime(end_time, start_time, units = "secs"))
  ))
  
  print(paste("Finished size:", sz))
}

print(results)


dfdata$outcome <- as.factor(dfdata$outcome)

sizes <- c(100, 1000, 10000, 100000, 1000000, 10000000)
results <- data.frame(Size = integer(), Accuracy = numeric(), Time = numeric())

set.seed(123)

for (sz in sizes) {
  # Sample the data
  dssample <- dfdata %>%
    sample_n(size = sz, replace = FALSE)
  
  # Set up trainControl for 5-fold CV
  trctrl <- trainControl(method = "cv", number = 5)
  
  # Train model and time it
  start_time <- Sys.time()
  model <- train(
    outcome ~ ., 
    data = dssample,
    method = "xgbTree",
    trControl = trctrl,
    verbose = 0
  )
  end_time <- Sys.time()
  
  # Get best accuracy from cross-validation
  acc <- max(model$results$Accuracy, na.rm = TRUE)
  
  # Save results
  results <- rbind(results, data.frame(
    Size = sz,
    Accuracy = acc,
    Time = as.numeric(difftime(end_time, start_time, units = "secs"))
  ))
  
  print(paste("Finished size:", sz))
}

print(results)