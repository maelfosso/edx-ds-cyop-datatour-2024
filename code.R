
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
submission <- read.csv(LOCAL_SUBMISSION_URL)

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
train <- train %>% group_by(country, lat, lon) %>%
  mutate(
    taux_adoption_energies_renouvelables = ifelse(
      is.na(taux_adoption_energies_renouvelables), 
      mean(taux_adoption_energies_renouvelables, na.rm = TRUE),
      taux_adoption_energies_renouvelables
    )
  ) %>%
  ungroup() %>% as.data.frame()
train %>% group_by(country, lat, lon) %>% summarize(na = sum(is.na(taux_adoption_energies_renouvelables)))
sum(is.na(train)) # -> there is no NA values anymore

#### CATEGORICAL VARIABLES
# encoding transform the categorical features
# number of labels per categorical feature
bits_per_var <- train %>% select(categorical_features) %>% gather() %>%
  group_by(key) %>%
  summarize(b = ceiling(log2(length(unique(value)))))

label_encoding <- function(series) {
  return(as.integer(factor(series)))
}

convert_to_binary <- function(series, n) {
  s <- as.data.frame(do.call(rbind, as.binary(series, n = n))) %>%
          mutate(across(everything(), ~+as.logical(.x)))
  return(s)
}

binary_encoding <- function(data, name) {
  v <- label_encoding(data[, name]) - 1
  n <- bits_per_var %>% filter(key == name) %>% pull(b)
  dfv <- convert_to_binary(v, n)
  colnames(dfv) <- sapply(1:n, function(x) paste(name, x, sep = "_"))
  
  return(dfv)
}

# X.categorical_features <- do.call(cbind, lapply(categorical_features, binary_encoding))

##### NUMERICAL FEATURES
# from one of the chart we did above, we've notice that none of the numerical variable had a bell-shape histogram
# the first thing we will do will be to transform the skewed histogram to a one with bell-shape value

# we are going to normalize them using preProcess from caret package
data.num <- train %>% select(-categorical_features) %>% select(-c(demande_energetique_projectee))
train %>% select(-categorical_features) %>% gather() %>% 
  ggplot(aes(value)) + geom_histogram() + facet_wrap(~key, nrow = 5, ncol = 4, scales = "free_x")

preProc.YJ <- preProcess(data.num, method = c("YeoJohnson"))
data.num.yj <- predict(preProc.YJ, data.num)
data.num.yj %>% gather() %>% 
  ggplot(aes(value)) + geom_histogram() + facet_wrap(~key, nrow = 5, ncol = 4, scales = "free_x")

preProc.BC <- preProcess(data.num, method = c("BoxCox"))
data.num.bc <- predict(preProc.BC, data.num)
data.num.bc %>% gather() %>% 
  ggplot(aes(value)) + geom_histogram() + facet_wrap(~key, nrow = 5, ncol = 4, scales = "free_x")

preProc.CS <- preProcess(data.num, method = c("center", "scale"))
data.num.cs <- predict(preProc.CS, data.num)
data.num.cs %>% gather() %>% 
  ggplot(aes(value)) + geom_histogram() + facet_wrap(~key, nrow = 5, ncol = 4, scales = "free_x")

preProc.CS <- preProcess(data.num, method = c("center", "scale", "YeoJohnson", "BoxCox"))
data.num.cs <- predict(preProc.CS, data.num)
data.num.cs %>% gather() %>% 
  ggplot(aes(value)) + geom_histogram() + facet_wrap(~key, nrow = 5, ncol = 4, scales = "free_x")
head(data.num.cs)

preprocessing_data <- function(data) {
  data.num <- data %>% select(-categorical_features) %>% select(-c(demande_energetique_projectee))
  data.cat <- data %>% select(categorical_features)

  preProc <- preProcess(data.num, method = c("center", "scale", "YeoJohnson", "BoxCox"))
  X.numerical_features <- predict(preProc, data.num)
  
  X.categorical_features <- do.call(cbind, lapply(categorical_features, function(x) binary_encoding(data.cat, x)))
  
  return(cbind(X.numerical_features, X.categorical_features))
}

# Train data for machine learning
train.data.num <- train %>% select(-categorical_features) %>% select(-c(demande_energetique_projectee))
train.data.cat <- train %>% select(categorical_features)

preProc <- preProcess(train.data.num, method = c("center", "scale", "YeoJohnson", "BoxCox"))
train.X.numerical_features <- predict(preProc, train.data.num)
train.X.categorical_features <- do.call(cbind, lapply(categorical_features, function(x) binary_encoding(train.data.cat, x)))

X.train <- cbind(train.X.numerical_features, train.X.categorical_features)
Y.train <- train %>% select(demande_energetique_projectee)
XY.train <- cbind(X.train, Y.train)

# Test data for machine learning
test.data.num <- test %>% select(-categorical_features) %>% select(-c(demande_energetique_projectee))
test.data.cat <- test %>% select(categorical_features)

test.X.numerical_features <- predict(preProc, test.data.num)
test.X.categorical_features <- do.call(cbind, lapply(categorical_features, function(x) binary_encoding(test.data.cat, x)))

X.test <- cbind(test.X.numerical_features, test.X.categorical_features)
Y.test <- test %>% select(demande_energetique_projectee)
XY.test <- cbind(X.test, Y.test)

# Submission data for machine learning
submission.data.num <- submission %>% select(-categorical_features) # %>% select(-c(demande_energetique_projectee))
submission.data.cat <- submission %>% select(categorical_features)

submission.X.numerical_features <- predict(preProc, submission.data.num)
submission.X.categorical_features <- do.call(cbind, lapply(categorical_features, function(x) binary_encoding(submission.data.cat, x)))

X.submission <- cbind(submission.X.numerical_features, submission.X.categorical_features)


########
# MACHINE LEARNING
#######

set.seed(1)

# set-up cross-validation
cv <- trainControl(
  method = "cv",
  number = 10,
  verboseIter = TRUE
)

# KNN
# Find the best K (min RMSE) from cross-validation on training data
tuneGrid <- expand.grid(
  k = seq(3, 50, by = 1)
)
model.knn <- caret::train(
  demande_energetique_projectee ~ .,
  data = XY.train,
  method = 'knn',
  trControl = cv,
  tuneGrid = tuneGrid,
  verbose = TRUE
)
model.knn
plot(model.knn)
