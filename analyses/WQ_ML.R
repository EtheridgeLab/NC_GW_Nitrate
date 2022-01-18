---
title: "Groundwater Quality - Machine Learning Regression"
author: "Randall Etheridge, Jake Hochard, Ariane Peralta, Tom Vogel"
---

#PC - set WD manually by Session -> Set Working Directory -> Choose Directory...
setwd("/") #Add the path for your PC here

#load req'd packages
library("tidymodels")
library("tidyverse")
library("workflows")
library("tune")
library("doFuture")
library("doRNG")

set.seed(123) #set seed for repeatability

#set up for parallel processing
registerDoFuture()
plan(multisession, gc = TRUE)

#load data file 
master <- read_csv("NO3_precip_temp_region_SA.csv")
dim(master)

#Remove #'s in one of the next three sections to determine which region to develop models for

#use the whole state
#commast <- master %>%
#  select(-region, -sa)

#choose region
#commast <- master %>%
#  filter(region == "M") %>%  #"CP" for Coastal Plain; "P" for Piedmont; "M" for Mountains
#  select(-region, -sa)

#choose SA
#commast <- master %>%
#  filter(sa == "C") %>% #"SA" for animal production area; "C" for control
#  select(-region, -sa)

commast$nitrate[commast$nitrate == 0] <- 0.5
#commast$nitrate <- log(commast$nitrate)

registerDoRNG(123)
# split the data into training (80%) and testing (20%)
commast_split <- initial_split(commast,prop = 4/5)

# extract training and testing sets
commast_train <- training(commast_split)
commast_test <- testing(commast_split)

# create CV object from training data
commast_cv <- vfold_cv(commast_train, v = 5)

# Define a recipe http://www.rebeccabarter.com/blog/2020-03-25_machine_learning/
ml_recipe <-
  #which consists of the formula (outcome ~ predictors)
  recipe(nitrate ~ ., data = commast) %>%
  step_normalize(all_predictors())

#Specify the model
rf_model <- 
  # specify that the model is a random forest
  # https://www.rdocumentation.org/packages/parsnip/versions/0.0.0.9001/topics/rand_forest
  rand_forest() %>%
  set_args(mtry = tune(),trees = tune()) %>%
  # select the engine/package that underlies the model
  set_engine("ranger", importance = "permutation") %>%
  # choose either the continuous regression or binary classification mode
  set_mode("regression")

# set the workflow
rf_workflow <- workflow() %>%
  # add the recipe
  add_recipe(ml_recipe) %>%
  # add the model
  add_model(rf_model)

# specify which values want to try
rf_grid <- expand.grid(mtry = c(1, 2, 3), trees = c(500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500))

# extract results
registerDoRNG(123)
start_time <- Sys.time()
rf_tune_results <- rf_workflow %>%
  tune_grid(resamples = commast_cv, #CV object
            grid = rf_grid, # grid of values to try
            metrics = metric_set(rmse, rsq) # metrics we care about
  )

end_time <- Sys.time()

# print results
met <- rf_tune_results %>%
  collect_metrics()

end_time - start_time

autoplot(rf_tune_results, metric = "rmse")
autoplot(rf_tune_results, metric = "rsq")

param_final <- rf_tune_results %>%
  select_best(metric = "rsq")
param_final

rf_workflow <- rf_workflow %>%
  finalize_workflow(param_final)

rf_fit <- rf_workflow %>%
  # fit on the training set and evaluate on test set
  last_fit(commast_split, metrics = metric_set(rmse, rsq))
rf_fit

test_performance <- rf_fit %>% collect_metrics()
test_performance

# generate predictions from the test set
test_predictions <- rf_fit %>% collect_predictions()
test_predictions

final_model <- fit(rf_workflow, commast)

ranger_obj <- pull_workflow_fit(final_model)$fit
ranger_obj

df <- as.data.frame(ranger_obj$variable.importance)
write.csv(df, "ML_C_082321.csv")