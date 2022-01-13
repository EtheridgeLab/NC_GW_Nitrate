---
title: "Groundwater Quality - State Time Series Regression"
author: "Randall Etheridge, Jake Hochard, Ariane Peralta, Tom Vogel"
---

#PC - set WD manually by Session -> Set Working Directory -> Choose Directory...
setwd("C:/Users/etheridgej15/OneDrive - East Carolina University/Research/EPA/CAFOs_2016/Data/")
#setwd("R:/EPA/Data/")
#setwd("U:/Research/EPA/Data/")

#load req'd packages
library("tidymodels")
library("tidyverse")
library("workflows")
library("tune")
library("doFuture")
library("doRNG")

set.seed(123)

all_cores <- parallel::detectCores(logical = FALSE)

registerDoFuture()
cl <- parallel::makeCluster(all_cores)
plan(future::cluster, workers = cl)
#plan(multisession, gc = TRUE)

#load data file 
master <- read_csv("NO3_precip_temp_region_SA.csv")
dim(master)

#do not choose region
commast <- master %>%
  select(-region, -sa)

#choose region
#commast <- master %>%
#  filter(region == "M") %>%
#  select(-region, -sa)

#choose SA
#commast <- master %>%
#  filter(sa == "C") %>%
#  select(-region, -sa)

#remove samples with no temperature and rainfall data
commast <- commast[!is.na(commast[2]), ]
commast <- commast[!is.na(commast[1]), ]

commast <- commast %>%
  mutate(nitrate = case_when(nitrate != 0 ~ "1", TRUE ~ "0"))

registerDoRNG(123)
# split the data into trainng (80%) and testing (20%)
commast_split <- initial_split(commast,prop = 4/5)

# extract training and testing sets
commast_train <- training(commast_split)
commast_test <- testing(commast_split)

# create CV object from training data
commast_cv <- vfold_cv(commast_train, v=5, repeats = 5)

# Define a recipe http://www.rebeccabarter.com/blog/2020-03-25_machine_learning/
GL_recipe <-
  #which consists of the formula (outcome ~ predictors)
  recipe(nitrate ~ ., data = commast)

#Specify the model
GL_model <- 
  logistic_reg() %>%
  # select the engine/package that underlies the model
  set_engine("glmnet") %>%
  # choose either the continuous regression or binary classification mode
  set_mode("classification")

# set the workflow
GL_workflow <- workflow() %>%
  # add the recipe
  add_recipe(GL_recipe) %>%
  # add the model
  add_model(GL_model)

# fit model
registerDoRNG(123)
GL_fit_rs <- fit_resamples(GL_workflow, commast_cv)

end_time <- Sys.time()

# print results
met <- GL_tune_results %>%
  collect_metrics()

end_time - start_time

autoplot(GL_tune_results, metric = "accuracy")
autoplot(GL_tune_results, metric = "kap")

param_final <- GL_tune_results %>%
  select_best(metric = "kap")
param_final

GL_workflow <- GL_workflow %>%
  finalize_workflow(param_final)

GL_fit <- GL_workflow %>%
  # fit on the training set and evaluate on test set
  last_fit(commast_split, metrics = metric_set(accuracy, kap))
GL_fit

test_performance <- GL_fit %>% collect_metrics()
test_performance

# generate predictions from the test set
test_predictions <- rf_fit %>% collect_predictions()
test_predictions

final_model <- fit(GL_workflow, commast)

GLM_obj <- final_model %>%
  pull_workflow_fit() %>%
  pluck("fit") %>%
  coef(s = param_final$penalty) %>%
  tidy()
GLM_obj

df <- as.data.frame(GLM_obj)
write_csv(df, "GLC_Whole_111220.csv")

#create dataframe with column headings of temperature and rainfall data
colna <- as.data.frame(colnames(commast[67:dim(commast)[2]]))
colnames(colna) <- c( 'header') #rename column

#separate column names strings into 4 columns and assign number to each row
colna2<-str_split_fixed(colna$header, "_", 4) #split column headings into different variables
colna <-cbind.data.frame(colna,colna2) #combine original column names with split column names
colnames(colna) <- c( 'header', "measure", "dl", "lag", "math") #rename column
colna <- within(colna, rm(dl)) #remove unneeded column
colna$te <- 1:nrow(colna)

#figure out how many unique values there are in each column 
#length(unique(colna$measure)) #Determine number of unique values #measure(4)
#length(unique(colna$lag)) #Determine number of unique values #lag(36)
#length(unique(colna$math)) #Determine number of unique values  #math(4)

#figure out how to match temp and ppt column headings with same description
pptna <- colna[(colna$measure=="ppt"),] #create dataframe with all precip variables
tempna <- colna[!(colna$measure=="ppt"),] #create dataframe with all temp variables
modlist <- merge(tempna,pptna,by=c("lag","math")) #merge precip and temp dataframes based on lag and max/min/mean

#Add month and yearly columns
#commast$month <- as.numeric(format(commast$sample_date,'%m'))
#commast$year <- as.numeric(format(commast$sample_date,'%Y'))

#Rename month and yearly columns
#rename(commast, c("sample_date2"="month", "sample_date1"="year"))
names(commast)[names(commast)=="sample_date2"] <- "month"
names(commast)[names(commast)=="sample_date1"] <- "year"

#Choose region for analysis - not running any of these lines runs the analysis for the whole state
commast <- subset(commast, county == "northampton" | county == "halifax" | county == "nash" |
                    county == "johnston" | county == "harnett" | county == "cumberland" |
                    county == "hoke" | county == "gates" | county == "currituck" |
                    county == "camden" | county == "scotland" | county == "hertford" |
                    county == "pasquotank" | county == "perquimans" | county == "chowan" |
                    county == "bertie" | county == "edgecombe" | county == "martin" |
                    county == "washington" | county == "tyrrell" | county == "dare" |
                    county == "wilson" | county == "pitt" | county == "beaufort" |
                    county == "hyde" | county == "wilson" | county == "wayne" |
                    county == "greene" | county == "lenoir" | county == "craven" |
                    county == "pamlico" | county == "sampson" | county == "duplin" |
                    county == "jones" | county == "carteret" | county == "onslow" |
                    county == "robeson" | county == "bladen" | county == "pender" |
                    county == "columbus" | county == "brunswick" | county == "new hanover")
commast <- subset(commast, county == "surry" | county == "stokes" | county == "rockingham" |
                    county == "caswell" | county == "person" | county == "granville" |
                    county == "vance" | county == "warren" | county == "franklin" |
                    county == "yadkin" | county == "forsyth" | county == "guilford" |
                    county == "alamance" | county == "orange" | county == "durham" |
                    county == "alexander" | county == "catawba" | county == "iredell" |
                    county == "davie" | county == "rowan" | county == "davidson" |
                    county == "randolph" | county == "chatham" | county == "wake" |
                    county == "lee" | county == "cleveland" | county == "lincoln" |
                    county == "gaston" | county == "mecklenburg" | county == "cabarrus" |
                    county == "union" | county == "stanly" | county == "anson" |
                    county == "montgomery" | county == "richmond" | county == "moore")
commast <- subset(commast, county == "ashe" | county == "alleghany" | county == "watauga" |
                    county == "wilkes" | county == "avery" | county == "caldwell" |
                    county == "mitchell" | county == "yancey" | county == "mcdowell" |
                    county == "burke" | county == "madison" | county == "buncombe" |
                    county == "haywood" | county == "swain" | county == "graham" |
                    county == "cherokee" | county == "clay" | county == "macon" |
                    county == "jackson" | county == "transylvania" | county == "henderson" |
                    county == "polk" | county == "rutherford")

#Nitrate
NO3 <- commast[!is.na(commast$nitrate), ] #separate samples with nitrate values
outcome <- "nitrate"
#NO3m <- lm(nitrate ~ tmean_lag_0_max + ppt_lag_0_max + tmean_lag_0_max*ppt_lag_0_max,NO3)
#summary(NO3m)
#summary(NO3m)$adj.r.squared
#summary(NO3m)$coefficients

#develop method to automatically run regressions for the selected combinations
#develop method to extract needed regression results
results<-as.data.frame(matrix(NA,dim(modlist)[1],17)) #create dataframe to record linear regression results
colnames(results) <- c("T","P","Intercept","T_C","P_C","M_C","Y_C","C_C","I_C","Intercept_p","T_p","P_p","M_p","Y_p","C_p","I_p","R^2")

#headers <- c(as.character(modlist$header.x[2]),as.character(modlist$header.y[2]))
#inter <- paste(headers, collapse = "*")
#headers <- c(headers,inter,"month","year","countyid")
#f <- as.formula(paste(outcome, paste(headers, collapse = " + "), sep = " ~ "))
#NO3m <- lm(f, NO3)
#summary(NO3m)

#NO3 <- NO3[format(NO3$sample_date,'%m') >="04", ] #Choose months for analysis
#NO3 <- NO3[format(NO3$sample_date,'%m') <="10", ] #Choose months for analysis

for (i in 1:(dim(modlist)[1])){
  headers <- c(as.character(modlist$header.x[i]),as.character(modlist$header.y[i]))
  inter <- paste(headers, collapse = "*")
  headers <- c(headers,inter,"month","year","countyid")
  f <- as.formula(paste(outcome, paste(headers, collapse = " + "), sep = " ~ "))
  NO3m <- lm(f, NO3)
  results[i,1] <- as.character(modlist$header.x[i])
  results[i,2] <- as.character(modlist$header.y[i])
  results[i,3] <- summary(NO3m)$coefficients[1]
  results[i,4] <- summary(NO3m)$coefficients[2]
  results[i,5] <- summary(NO3m)$coefficients[3]
  results[i,6] <- summary(NO3m)$coefficients[4]
  results[i,7] <- summary(NO3m)$coefficients[5]
  results[i,8] <- summary(NO3m)$coefficients[6]
  results[i,9] <- summary(NO3m)$coefficients[7]
  results[i,10] <- summary(NO3m)$coefficients[22]
  results[i,11] <- summary(NO3m)$coefficients[23]
  results[i,12] <- summary(NO3m)$coefficients[24]
  results[i,13] <- summary(NO3m)$coefficients[25]
  results[i,14] <- summary(NO3m)$coefficients[26]
  results[i,15] <- summary(NO3m)$coefficients[27]
  results[i,16] <- summary(NO3m)$coefficients[28]
  results[i,17] <- summary(NO3m)$r.squared
}

write.csv(results,file="NO3_MYC_control_M.csv",row.names=FALSE) #Export results to .csv file

#plot(modlist$lag,results$T_p)
#plot(modlist$lag,results$P_p)
#plot(modlist$lag,results$I_p)

#PLSR
FP <- NO3[22:dim(NO3)[2]]
fit<-plsr(NO3$nitrate~data.matrix(FP),ncomp=20,validation="CV")
summary(fit)
