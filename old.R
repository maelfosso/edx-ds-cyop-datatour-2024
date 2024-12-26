

# train %>% group_by(country, lat, lon) %>%
# mutate(
#   taux_adoption_energies_renouvelables = ifelse(
#     is.na(taux_adoption_energies_renouvelables), 
#     mean(taux_adoption_energies_renouvelables, na.rm = TRUE),
#     taux_adoption_energies_renouvelables
#   )
# ) %>%
# ungroup() %>% as.data.frame()
# train %>% group_by(country, lat, lon) %>% summarize(na = sum(is.na(taux_adoption_energies_renouvelables)))
# sum(is.na(train)) # -> there is no NA values anymore

#####

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

##########


# we are going to normalize them using preProcess from caret package

basic_recipe <-
  basic_recipe %>%
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
