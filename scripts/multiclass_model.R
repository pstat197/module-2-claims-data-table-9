library(tidyverse)
library(tm)
library(SnowballC)
library(e1071)
library(text2vec)

load("data/claims-clean.RData")
source("scripts/preprocessing.R")

df <- claims_clean

df$mclass <- droplevels(df$mclass)
labels <- df$mclass
texts  <- df$text_clean

# create corpus
corpus <- VCorpus(VectorSource(texts))

corpus <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("en")) %>%
  tm_map(stripWhitespace)

# build TF–IDF matrix
dtm <- DocumentTermMatrix(corpus, control = list(weighting = weightTfIdf))

dtm <- removeSparseTerms(dtm, 0.99)

X <- as.matrix(dtm)
y <- as.factor(labels)       # multi-class label

# split train-test sets
set.seed(14)
train_idx <- sample(seq_len(nrow(X)), size = 0.8 * nrow(X))

X_train <- X[train_idx, ]
X_test  <- X[-train_idx, ]
y_train <- y[train_idx]
y_test  <- y[-train_idx]

# train multi-class SVM
svm_model <- svm(
  x = X_train,
  y = y_train,
  kernel = "linear",
  cost = 1,
  scale = TRUE,
  probability = TRUE
)

# evaluate internal performance
pred <- predict(svm_model, X_test)
mean(pred == y_test)
table(pred, y_test)

svm_final <- svm(
  x = X,
  y = y,
  kernel = "linear",
  cost = 1,
  scale = TRUE,
  probability = TRUE
)

# save svm model into results for deployment
save(svm_final, file = "results/svm_mclass_model.RData")

load("data/claims-test.RData")   # has raw HTML but no labels

claims_test_clean <- parse_data_safe_na(claims_test)
#dim(claims_test_clean)
#names(claims_test_clean)
test_df <- claims_test_clean

# replace NAs with empty strings to preserve number of rows as claims_test without impacting performance
test_df$text_clean[is.na(test_df$text_clean)] <- ""

# build corpus and dtm using the *same vocabulary*
test_corpus <- VCorpus(VectorSource(test_df$text_clean)) %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("en")) %>%
  tm_map(stripWhitespace)

test_dtm <- DocumentTermMatrix(test_corpus, control = list(dictionary = Terms(dtm)))
X_new <- as.matrix(test_dtm)

# predict
mclass_pred <- predict(svm_final, X_new)

pred_df <- data.frame(
  .id = claims_test$.id,
  mclass.pred = mclass_pred
)

save(pred_df, file = "results/svm_mclass_predictions.RData")
head(pred_df)
