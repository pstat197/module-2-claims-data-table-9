library(tidyverse)
library(text2vec)
library(glmnet)

# Load training data
load("data/claims-clean-example.RData")
df <- claims_clean

# Binary outcome and dropping extra factor levels
df$bclass <- droplevels(df$bclass)
labels <- df$bclass
texts  <- df$text_clean

# Tokenization + TF-IDF with vocabulary pruning
it <- itoken(texts,
             tokenizer = word_tokenizer,
             progressbar = TRUE)

#lower threshold and removing common words, words that show up less than 3 times and removes words that appear in less than 50% of documents
vocab <- create_vocabulary(it) %>%
  prune_vocabulary(term_count_min = 3,    
                   doc_proportion_max = 0.5) 

vectorizer <- vocab_vectorizer(vocab)
dtm <- create_dtm(it, vectorizer)

tfidf <- TfIdf$new()
dtm_tfidf <- tfidf$fit_transform(dtm)

# LASSO logistic regression
y <- as.numeric(labels == levels(labels)[2])

# 10-fold cross-validation
set.seed(123)
cvfit <- cv.glmnet(dtm_tfidf, y,
                   family = "binomial",
                   alpha = 1,
                   type.measure = "class",
                   nfolds = 10)  

model <- cvfit

# Load test data + preprocessing
load("data/claims-test.RData")
df_test <- claims_test

it_test <- itoken(df_test$text_tmp,
                  tokenizer = word_tokenizer,
                  progressbar = TRUE)

dtm_test <- create_dtm(it_test, vectorizer)
dtm_test_tfidf <- tfidf$transform(dtm_test)

# Predictions
test_pred_prob <- predict(
  model, dtm_test_tfidf,
  s = "lambda.min",
  type = "response"
)[,1]

test_pred_class <- ifelse(
  test_pred_prob > 0.5,
  levels(labels)[2],
  levels(labels)[1]
)
pred_df <- tibble(
  .id = df_test$.id,
  bclass.pred = test_pred_class
)

save(model, file = "results/model-group9-binary.RData")
save(pred_df, file = "results/preds-group9.RData")

# Results
train_prob <- predict(model, dtm_tfidf, s = "lambda.min", type = "response")[,1]
train_pred <- ifelse(train_prob > 0.5, 1, 0)
true_y <- y

# Metrics
accuracy    <- mean(train_pred == true_y)
sensitivity <- mean(train_pred[true_y == 1] == 1)
specificity <- mean(train_pred[true_y == 0] == 0)

accuracy
sensitivity
specificity