
#####
# SETUP
####

if (!require(devtools)) install.packages("devtools")
library(devtools)

if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
if (!require(preprocessCore)) BiocManager::install("preprocessCore")
if (!require("TidyDensity")) install.packages("TidyDensity")
if (!require(binaryLogic)) install_github("d4ndo/binaryLogic")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(GGally)) install.packages("GGally")

library(tidyverse)
library(preprocessCore)
library(TidyDensity)
library(binaryLogic)
library(ggplot2)
library(GGally)
library(caret)
library(tidymodels)

library(doParallel)
library(future)
plan(multisession)

####
# LOAD DATASET
####

REMOTE_TRAIN_URL = "https://raw.githubusercontent.com/dataafriquehub/energy_data/refs/heads/main/train.csv"
REMOTE_TEST_URL = "https://raw.githubusercontent.com/dataafriquehub/energy_data/refs/heads/main/test.csv"
REMOTE_SUBMISSION_URL = "https://raw.githubusercontent.com/dataafriquehub/energy_data/refs/heads/main/submission.csv"

# Create directory for keeping datasets if it doesn't exists
DATASETS_DIR <- "./datasets"
if (!file.exists(DATASETS_DIR)) {
  dir.create(path = DATASETS_DIR, showWarnings = FALSE, recursive = TRUE)
}


LOCAL_TRAIN_URL = file.path(DATASETS_DIR, "train.csv")
LOCAL_TEST_URL = file.path(DATASETS_DIR, "test.csv")
LOCAL_SUBMISSION_URL = file.path(DATASETS_DIR, "submission.csv")

# Download the CSV files
csv_files <- list.files(path = DATASETS_DIR, pattern = "*.csv")
if (length(csv_files) < 3) {
  download.file(REMOTE_TRAIN_URL, LOCAL_TRAIN_URL)
  download.file(REMOTE_TEST_URL, LOCAL_TEST_URL)
  download.file(REMOTE_SUBMISSION_URL, LOCAL_SUBMISSION_URL)
}

# Read CSV
train <- read.csv(LOCAL_TRAIN_URL)
test <- read.csv(LOCAL_TEST_URL)
# submission <- read.csv(LOCAL_SUBMISSION_URL)

#####
# EXPLORATORY DATA ANALYSIS
#####

# the size of the different dataset
dim(train)
dim(test)
dim(submission)

### Let's focus on train data sets

## List the different columns
colnames(train)

## describe it
summary(train)
# -> categorical: country, types_sols, habit_de_mariage
# -> numerical: all others variables

## Variables with NA
colSums(is.na(train))
# -> variables with NA: taux_adoption_energies_renouvelables (15136)

# Explore: Country
# number of unique values
length(unique(train$country)) # -> 53 countries
prop.table(table(train$country)) * 100 # -> percentage of rows per country: around 2% per country
ggplot(data = train, aes(x = country)) + geom_bar() + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# Explore: Lat
length(unique(train$lat)) # -> 53 different values same than the number of country
ggplot(data = train, aes(x = lat)) + geom_histogram()

# Explore: Lon
length(unique(train$lon)) # -> 53 different values same than the number of country
ggplot(data = train, aes(x = lon)) + geom_histogram()

# same number of lat, lon, country
# let's check if they are all 3 uniques
train %>% distinct(country, lat, lon) %>% nrow # -> they. are 53 unique rows
# it means that we will have to choose between country and (lat, lon) couple

# Explore: population
length(unique(train$population)) # -> 53 also, so it's linked to the country
ggplot(data = train, aes(x = population)) + geom_histogram() # -> skewed at right
ggplot(data = train, aes(x = population)) + geom_boxplot()

# Explore: taux_ensoleillement
length(unique(train$taux_ensoleillement)) # -> there are different values
ggplot(data = train, aes(x = taux_ensoleillement)) + geom_histogram() # -> bell shape not present
ggplot(data = train, aes(x = taux_ensoleillement)) + geom_boxplot() # -> Good repartition

# Explore: demande_energetique_actuelle
length(unique(train$demande_energetique_actuelle)) # -> different values / variables
ggplot(data = train, aes(x = demande_energetique_actuelle)) + geom_histogram()
ggplot(data = train, aes(x = demande_energetique_actuelle)) + geom_boxplot() # -> almost good variabilite in the data

# to go fast in the EDA, let's plot the different variables into on chart
categorical_features <- c("country", "types_sols", "habit_de_mariage")
train %>% select(-categorical_features) %>% gather() %>% 
  ggplot(aes(value)) + geom_histogram() + facet_wrap(~key, nrow = 5, ncol = 4, scales = "free_x")
# -> nombre_animaux_domestiques, potentiel_investissement are categorical variables

categorical_features <- c("country", "types_sols", "habit_de_mariage", "nombre_animaux_domestiques", "potentiel_investissement")
train %>% select(-categorical_features) %>% gather() %>% 
  ggplot(aes(value)) + geom_histogram() + facet_wrap(~key, nrow = 5, ncol = 4, scales = "free_x")
# we have a global view of the data, 
# - none of the features has a histogram with bell shape (transformation is needed)
# - 
train %>% select(-categorical_features) %>% gather() %>% 
  ggplot(aes(value)) + geom_boxplot() + facet_wrap(~key, nrow = 5, ncol = 4, scales = "free_x")
# we have a global view of the numerical data
# - most of the data seems not having an outlier
# - population field has an outlier

# Visualize the correlation between all the variable and the `demande_energetique_projectee` 
# ggpairs(train %>% select(-categorical_features))
# we can notice that the only correlation that exists is the one between `demande_energetique_actuelle` and `demande_energetique_projetee`
# all the other variable are independant to the response (`demande_energetique_projetee`)

numerical_features <- setdiff(train %>% select(-demande_energetique_projectee) %>% names(), categorical_features)
numerical_features


# Log10 demande_energetique_projectee
train$demande_energetique_projectee = log10(train$demande_energetique_projectee)

#####
# FEATURE ENGINEERING
####

#### MISSING DATA
# Let's handle missing data
# number of na per country
train %>% group_by(country, lat, lon) %>% summarize(na = sum(is.na(taux_adoption_energies_renouvelables)))

# percentage of na per country
train %>% group_by(country, lat, lon) %>% 
  summarize(na = sum(is.na(taux_adoption_energies_renouvelables)),
            n = n(),
            na_ratio = na/n,
            .groups = 'drop')
# almost same number of na per country

# let's replace NA by the mean of `taux_adoption_energies_renouvelables` in the current group
basic_recipe <- 
  recipe(demande_energetique_projectee ~ ., data = train) %>%
  step_impute_mean(taux_adoption_energies_renouvelables)
  # step_impute_linear(taux_adoption_energies_renouvelables, impute_with = imp_vars(country, lat, lon))

#### CATEGORICAL VARIABLES
# encoding transform the categorical features
# number of labels per categorical feature
basic_recipe <- 
  basic_recipe %>%
  step_mutate(
    nombre_animaux_domestiques = as.character(nombre_animaux_domestiques), 
    potentiel_investissement = as.character(potentiel_investissement)
  ) %>%
  step_string2factor(c("country", "types_sols", "habit_de_mariage", "nombre_animaux_domestiques", "potentiel_investissement"))

one_hot_recipe <- basic_recipe %>%
  step_dummy(categorical_features, one_hot = TRUE)

##### NUMERICAL FEATURES
# from one of the chart we did above, we've notice that none of the numerical variable had a bell-shape histogram
# the first thing we will do will be to transform the skewed histogram to a one with bell-shape value

one_hot_recipe <-
  one_hot_recipe %>%
  step_normalize(numerical_features) %>%
  step_YeoJohnson(numerical_features)

# basic_recipe <- 
#   basic_recipe %>%
#   step_BoxCox(setdiff(numerical_features, c("lat", "lon"))) %>%
#   step_normalize(numerical_features)

one_hot_recipe %>%
  prep() %>%
  # summary() %>% as.data.frame()
  juice() %>% is.na() %>% sum()

######
# Creating folds
######

set.seed(1)

folds <- vfold_cv(train, v = 10, strata = demande_energetique_projectee)
folds

######
# PARALLEL TUNING
######
tuning_parallel <- function() {
  all_cores <- parallel::detectCores(logical = FALSE)

  cl <- makePSOCKcluster(all_cores)
  registerDoParallel(cl)
}
tuning_parallel()
########
# MACHINE LEARNING
#######

tree_model <-
  decision_tree(tree_depth = tune(), min_n = tune(), cost_complexity = tune()) %>%
  set_engine('rpart') %>%
  set_mode('regression')

lr_model <-
  linear_reg() %>%
  set_engine('lm')

elastic_net_model <- 
  linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine("glmnet")

lasso_model <- 
  linear_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet")

ridge_model <- 
  linear_reg(penalty = tune(), mixture = 0) %>%
  set_engine("glmnet")

knn_model <-
  nearest_neighbor(neighbors = tune(), weight_func = tune(), dist_power = tune()) %>%
  set_engine('kknn') %>%
  set_mode('regression')

###### 
# WORKFLOW
######

control <- control_resamples(save_pred = TRUE, verbose = TRUE)

### Tune grid Tree
tree_grid <-
  grid_regular(
    cost_complexity(),
    tree_depth(),
    min_n(),
    levels = c(3, 5, 10)
  )

tree_wf <-
  workflow() %>%
  add_model(tree_model) %>%
  add_recipe(basic_recipe)

tree_res <- 
  tree_wf %>%
  tune_grid(
    resamples = folds,
    grid = tree_grid,
    metrics = metric_set(rmse),
    control = control
  )
tree_res %>% 
  collect_metrics() %>%
  arrange(mean) %>%
  print(n = 50)
tree_res %>%
  show_best(metric = "rmse")

best_tree <- tree_res %>%
  select_best(metric = "rmse")
best_tree
# A tibble: 1 × 4
# cost_complexity tree_depth min_n .config               
# <dbl>      <int> <int> <chr>                 
# 0.00000316          8    40 Preprocessor1_Model143

final_tree_wf <-
  tree_wf %>% 
  finalize_workflow(best_tree)
# ══ Workflow ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Preprocessor: Formula
# Model: decision_tree()
# 
# ── Preprocessor ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# demande_energetique_projectee ~ .
# 
# ── Model ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Decision Tree Model Specification (regression)
# 
# Main Arguments:
#   cost_complexity = 3.16227766016838e-06
# tree_depth = 8
# min_n = 31
# 
# Computational engine: rpart 

# Let's re-run the best model with CV to get a result to compare
final_tree_res <-
  final_tree_wf %>%
  fit_resamples(
    folds,
    metrics = metric_set(rmse),
    control = control
  )
final_tree_res %>%
  collect_metrics()
# A tibble: 1 × 6
# .metric .estimator   mean     n   std_err .config             
# <chr>   <chr>       <dbl> <int>     <dbl> <chr>               
# rmse    standard   0.0804    10 0.0000731 Preprocessor1_Model1

### Tune grid KNN

knn_grid <-
  grid_regular(
    neighbors(),
    weight_func(),
    dist_power(),
    levels = c(10, 5, 5)
  )

knn_wf <-
  workflow() %>%
  add_model(knn_model) %>%
  add_formula(demande_energetique_projectee ~ .)

knn_res <-
  knn_wf %>%
  tune_grid(
    resamples = folds,
    grid = knn_grid,
    metrics = metric_set(rmse)
  )
knn_res

#### Tune grid Linear Regression
lr_wf <-
  workflow() %>%
  add_model(lr_model) %>%
  add_recipe(basic_recipe)

lr_res <-
  lr_wf %>%
  fit_resamples(
    folds,
    metrics = metric_set(rmse),
    control = control
  )
lr_res %>%
  collect_metrics()
# A tibble: 1 × 6
# .metric .estimator  mean     n  std_err .config             
# <chr>   <chr>      <dbl> <int>    <dbl> <chr>               
# rmse    standard   0.161    10 0.000614 Preprocessor1_Model1

# Linear Regression seems to perform better than Tree
# Let's test Elastic Net, Losso and Ridge 

### elastic_net Tune
elastic_net_grid <-
  grid_regular(
    penalty(),
    mixture(),
    levels = c(15, 10)
  )

elastic_net_wf <-
  workflow() %>%
  add_model(elastic_net_model) %>%
  add_recipe(basic_recipe)

elastic_net_res <- 
  elastic_net_wf %>%
  tune_grid(
    resamples = folds,
    grid = elastic_net_grid,
    metrics = metric_set(rmse),
    control = control
  )
elastic_net_res %>% 
  collect_metrics() %>%
  arrange(mean) %>%
  print(n = 50)
elastic_net_res %>%
  show_best(metric = "rmse")

best_elastic_net <- elastic_net_res %>%
  select_best(metric = "rmse")
best_elastic_net
# A tibble: 1 × 3
# penalty mixture .config               
# <dbl>   <dbl> <chr>                 
# 0.00139   0.333 Preprocessor1_Model056

final_elastic_net_wf <-
  elastic_net_wf %>% 
  finalize_workflow(best_elastic_net)

# Let's re-run the best model with CV to get a result to compare
final_elastic_net_res <-
  final_elastic_net_wf %>%
  fit_resamples(
    folds,
    metrics = metric_set(rmse),
    control = control
  )
final_elastic_net_res %>%
  collect_metrics()
# A tibble: 1 × 6
# .metric .estimator  mean     n  std_err .config             
# <chr>   <chr>      <dbl> <int>    <dbl> <chr>               
# 1 rmse    standard   0.161    10 0.000619 Preprocessor1_Model1

### Ridge Tune
ridge_grid <-
  grid_regular(
    penalty(),
    levels = c(25)
  )

ridge_wf <-
  workflow() %>%
  add_model(ridge_model) %>%
  add_recipe(basic_recipe)

ridge_res <- 
  ridge_wf %>%
  tune_grid(
    resamples = folds,
    grid = ridge_grid,
    metrics = metric_set(rmse),
    control = control
  )
ridge_res %>% 
  collect_metrics() %>%
  arrange(mean) %>%
  print(n = 50)
ridge_res %>%
  show_best(metric = "rmse")

best_ridge <- ridge_res %>%
  select_best(metric = "rmse")
best_ridge
# A tibble: 1 × 2
# penalty .config              
# <dbl> <chr>                
# 1 0.0000000001 Preprocessor1_Model01

final_ridge_wf <-
  ridge_wf %>% 
  finalize_workflow(best_ridge)

# Let's re-run the best model with CV to get a result to compare
final_ridge_res <-
  final_ridge_wf %>%
  fit_resamples(
    folds,
    metrics = metric_set(rmse),
    control = control
  )
final_ridge_res %>%
  collect_metrics()
# A tibble: 1 × 6
# .metric .estimator  mean     n  std_err .config             
# <chr>   <chr>      <dbl> <int>    <dbl> <chr>               
# 1 rmse    standard   0.163    10 0.000703 Preprocessor1_Model1


### Lasso Tune
lasso_grid <-
  grid_regular(
    penalty(),
    levels = c(25)
  )

lasso_wf <-
  workflow() %>%
  add_model(lasso_model) %>%
  add_recipe(basic_recipe)

lasso_res <- 
  lasso_wf %>%
  tune_grid(
    resamples = folds,
    grid = ridge_grid,
    metrics = metric_set(rmse),
    control = control
  )
lasso_res %>% 
  collect_metrics() %>%
  arrange(mean) %>%
  print(n = 50)
lasso_res %>%
  show_best(metric = "rmse")

best_lasso <- lasso_res %>%
  select_best(metric = "rmse")
best_lasso
# A tibble: 1 × 2
# penalty .config              
# <dbl> <chr>                
# 0.0000000001 Preprocessor1_Model01

final_lasso_wf <-
  lasso_wf %>% 
  finalize_workflow(best_lasso)

# Let's re-run the best model with CV to get a result to compare
final_lasso_res <-
  final_lasso_wf %>%
  fit_resamples(
    folds,
    metrics = metric_set(rmse),
    control = control
  )
final_lasso_res %>%
  collect_metrics()
# A tibble: 1 × 6
# .metric .estimator  mean     n  std_err .config             
# <chr>   <chr>      <dbl> <int>    <dbl> <chr>               
# rmse    standard   0.161    10 0.000623 Preprocessor1_Model1

###########
# PCA
###########
pca_recipe <-
  basic_recipe %>%
  step_mutate(
    country = as.numeric(country)
  ) %>%
  step_normalize(country, lat, lon, population) %>%
  step_YeoJohnson(country, lat, lon, population) %>%
  step_pca(country, lat, lon, population, threshold = tune(), num_comp = tune()) %>%
  step_normalize(all_numeric()) %>%
  step_YeoJohnson(all_numeric()) %>%
  step_dummy(types_sols, habit_de_mariage, nombre_animaux_domestiques, potentiel_investissement , one_hot = TRUE)
  # step_normalize(
  #   taux_ensoleillement, demande_energetique_actuelle, capacite_installee_actuelle,
  #   duree_ensoleillement_annuel, cout_installation_solaire, proximite_infrastructures_energetiques,
  #   taux_adoption_energies_renouvelables, stabilite_politique, taux_acces_energie, niveau_urbanisation,
  #   emissions_co2_evitees, idh
  # ) %>%
  # step_YeoJohnson(
  #   taux_ensoleillement, demande_energetique_actuelle, capacite_installee_actuelle,
  #   duree_ensoleillement_annuel, cout_installation_solaire, proximite_infrastructures_energetiques,
  #   taux_adoption_energies_renouvelables, stabilite_politique, taux_acces_energie, niveau_urbanisation,
  #   emissions_co2_evitees, idh
  # )

# pca_recipe %>%
#   prep() %>% juice() %>% names()

### elastic_net Tune
elastic_net_grid <-
  parameters(
    num_comp(c(1, 9)),
    threshold(),
    penalty(),
    mixture()
  ) %>%
  grid_regular(levels = c(15, 10, 4, 10)) %>%
  arrange(num_comp, threshold, penalty, mixture)

elastic_net_wf <-
  workflow() %>%
  add_model(elastic_net_model) %>%
  add_recipe(pca_recipe)

elastic_net_res <- 
  elastic_net_wf %>%
  tune_grid(
    resamples = folds,
    grid = elastic_net_grid,
    metrics = metric_set(rmse),
    control = control
  )
elastic_net_res %>% 
  collect_metrics() %>%
  arrange(mean) %>%
  print(n = 50)
elastic_net_res %>%
  show_best(metric = "rmse")

best_elastic_net <- elastic_net_res %>%
  select_best(metric = "rmse")
best_elastic_net


final_elastic_net_wf <-
  elastic_net_wf %>% 
  finalize_workflow(best_elastic_net)

# Let's re-run the best model with CV to get a result to compare
final_elastic_net_res <-
  final_elastic_net_wf %>%
  fit_resamples(
    folds,
    metrics = metric_set(rmse),
    control = control
  )
final_elastic_net_res %>%
  collect_metrics()


