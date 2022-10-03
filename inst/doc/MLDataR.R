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

## ----ls_one-------------------------------------------------------------------
library(MLDataR)
library(dplyr)
library(ggplot2)
library(caret)
library(rsample)
library(varhandle)

data("long_stayers")
glimpse(long_stayers)


## ----ls_two-------------------------------------------------------------------
long_stayers <- long_stayers %>% 
  dplyr::mutate(stranded.label=factor(stranded.label)) %>% 
  dplyr::select(everything(), -c(admit_date))

cats <- select_if(long_stayers, is.character)
cat_dummy <- varhandle::to.dummy(cats$frailty_index, "frail_ind") 
#Converts the frailty index column to dummy encoding and sets a column called "frail_ind" prefix
cat_dummy <- cat_dummy %>% 
  as.data.frame() %>% 
  dplyr::select(-frail_ind.No_index_item) #Drop the field of interest
# Drop the frailty index from the stranded data frame and bind on our new encoding categorical variables
long_stayers <- long_stayers %>% 
  dplyr::select(-frailty_index) %>% 
  bind_cols(cat_dummy) %>% na.omit(.)

## ----ls_three-----------------------------------------------------------------
split <- rsample::initial_split(long_stayers, prop = 3/4)
train <- rsample::training(split)
test <- rsample::testing(split)

set.seed(123)
glm_class_mod <- caret::train(factor(stranded.label) ~ ., data = train,
                 method = "glm")
print(glm_class_mod)

## ----ls_four------------------------------------------------------------------
split <- rsample::initial_split(long_stayers, prop = 3/4)
train <- rsample::training(split)
test <- rsample::testing(split)

set.seed(123)
glm_class_mod <- caret::train(factor(stranded.label) ~ ., data = train, 
                 method = "glm")
print(glm_class_mod)

## ----ls_five------------------------------------------------------------------
preds <- predict(glm_class_mod, newdata = test) # Predict class
pred_prob <- predict(glm_class_mod, newdata = test, type="prob") #Predict probs

# Join prediction on to actual test data frame and evaluate in confusion matrix

predicted <- data.frame(preds, pred_prob)
test <- test %>% 
  bind_cols(predicted) %>% 
  dplyr::rename(pred_class=preds)

glimpse(test)

## ----ls_six-------------------------------------------------------------------
library(ConfusionTableR)
cm <- ConfusionTableR::binary_class_cm(test$stranded.label, test$pred_class, positive="Stranded")
cm$record_level_cm

library(OddsPlotty)
plotty <- OddsPlotty::odds_plot(glm_class_mod$finalModel,
                                title = "Odds Plot ",
                                subtitle = "Showing odds of patient stranded",
                                point_col = "#00f2ff",
                                error_bar_colour = "black",
                                point_size = .5,
                                error_bar_width = .8,
                                h_line_color = "red")
print(plotty)


