library(tidyverse)
library(text2vec)
library(glmnet)

# Load training data
load("data/claims-clean-example.RData")
df <- claims_clean

# Binary outcome
df$bclass <- droplevels(df$bclass)
labels <- df$bclass
texts  <- df$text_clean

# Tokenization + TF-IDF
it <- itoken(texts,
             tokenizer = word_tokenizer,
             progressbar = TRUE)

vocab <- create_vocabulary(it)
vectorizer <- vocab_vectorizer(vocab)

dtm <- create_dtm(it, vectorizer)

tfidf <- TfIdf$new()
dtm_tfidf <- tfidf$fit_transform(dtm)

# LASSO logistic regression
y <- as.numeric(labels == levels(labels)[2])  # convert factor → 0/1

set.seed(123)
cvfit <- cv.glmnet(dtm_tfidf, y,
                   family = "binomial",
                   alpha = 1,           # LASSO
                   type.measure = "class")

model <- cvfit

# Load TEST data + preprocess
load("data/claims-test.RData")
df_test <- claims_test

it_test <- itoken(df_test$text_tmp,
                  tokenizer = word_tokenizer,
                  progressbar = TRUE)

dtm_test <- create_dtm(it_test, vectorizer)
dtm_test_tfidf <- tfidf$transform(dtm_test)

# Predictions
test_pred_prob <- predict(model, dtm_test_tfidf, 
                          s = "lambda.min", 
                          type = "response")[,1]

test_pred_class <- ifelse(test_pred_prob > 0.5,
                          levels(labels)[2],
                          levels(labels)[1])

pred_df <- tibble(
  .id = df_test$.id,
  bclass.pred = test_pred_class
)

save(pred_df, file = "results/preds-group9.RData")

