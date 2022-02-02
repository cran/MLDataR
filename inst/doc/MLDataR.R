## ---- include = FALSE---------------------------------------------------------
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.height= 5, 
  fig.width=7
)

## ----setup, include = FALSE, echo=FALSE---------------------------------------
library(MLDataR)
library(dplyr)
library(ConfusionTableR)
library(parsnip)
library(rsample)
library(recipes)
library(ranger)
library(workflows)
library(caret)


## ----install_MLDataR----------------------------------------------------------
#install.packages(MLDataR)
library(MLDataR)


## ----thyroid_data-------------------------------------------------------------

glimpse(MLDataR::thyroid_disease)


## ----data_prep----------------------------------------------------------------
data("thyroid_disease")
td <- thyroid_disease
# Create a factor of the class label to use in ML model
td$ThryroidClass <- as.factor(td$ThryroidClass)
# Check the structure of the data to make sure factor has been created
str(td)

## ----remove_nulls-------------------------------------------------------------
# Remove missing values, or choose more advaced imputation option
td <- td[complete.cases(td),]
#Drop the column for referral source
td <- td %>%
   dplyr::select(-ref_src)


## ----splitting----------------------------------------------------------------
#Divide the data into a training test split
set.seed(123)
split <- rsample::initial_split(td, prop=3/4)
train_data <- rsample::training(split)
test_data <- rsample::testing(split)


## ----create_recipe------------------------------------------------------------
td_recipe <-
   recipe(ThryroidClass ~ ., data=train_data) %>%
   step_normalize(all_predictors()) %>%
   step_zv(all_predictors())

print(td_recipe)

## ----random_forest_model------------------------------------------------------
set.seed(123)
rf_mod <-
  parsnip::rand_forest() %>%
  set_engine("ranger") %>%
  set_mode("classification")



## ----creating_workflow--------------------------------------------------------
td_wf <-
   workflow() %>%
   workflows::add_model(rf_mod) %>%
   workflows::add_recipe(td_recipe)

print(td_wf)
# Fit the workflow to our training data
set.seed(123)
td_rf_fit <-
   td_wf %>%
   fit(data = train_data)
# Extract the fitted data
td_fitted <- td_rf_fit %>%
    extract_fit_parsnip()


## ----make_preds_and_evaluate--------------------------------------------------
# Predict the test set on the training set to see model performance
class_pred <- predict(td_rf_fit, test_data)
td_preds <- test_data %>%
    bind_cols(class_pred)
# Convert both to factors
td_preds$.pred_class <- as.factor(td_preds$.pred_class)
td_preds$ThryroidClass <- as.factor(td_preds$ThryroidClass)

str(td_preds)

# Evaluate the data with ConfusionTableR
cm <- binary_class_cm(td_preds$.pred_class,
                      td_preds$ThryroidClass,
                      positive="sick")




## ----modelling_preds----------------------------------------------------------
#View Confusion matrix
cm$confusion_matrix
#View record level
cm$record_level_cm


## ----diabetes-----------------------------------------------------------------
glimpse(MLDataR::diabetes_data)

## ----load_in_heart------------------------------------------------------------
data(heartdisease)
# Convert diabetes data to factor'
hd <- heartdisease %>%
 mutate(HeartDisease = as.factor(HeartDisease))
is.factor(hd$HeartDisease)

## ----dummy_encode-------------------------------------------------------------
# Get categorical columns
hd_cat <- hd  %>%
  dplyr::select_if(is.character)
# Dummy encode the categorical variables 
 cols <- c("RestingECG", "Angina", "Sex")
# Dummy encode using dummy_encoder in ConfusionTableR package
coded <- ConfusionTableR::dummy_encoder(hd_cat, cols, remove_original = TRUE)
coded <- coded %>%
     select(RestingECG_ST, RestingECG_LVH, Angina=Angina_Y,
     Sex=Sex_F)
# Remove column names we have encoded from original data frame
hd_one <- hd[,!names(hd) %in% cols]
# Bind the numerical data on to the categorical data
hd_final <- bind_cols(coded, hd_one)
# Output the final encoded data frame for the ML task
glimpse(hd_final)

