library(tidyverse)
library(text2vec)
library(glmnet)

# Load model
load("results/model-group9-binary.RData")

# Load test data
load("data/claims-test.RData")
df_test <- claims_test

it_test <- itoken(
  df_test$text_tmp,
  tokenizer = word_tokenizer,
  progressbar = TRUE
)

vectorizer <- model$glmnet.fit$dimnames$Terms 

dtm_test <- create_dtm(it_test, vocab_vectorizer(vocab))
dtm_test_tfidf <- tfidf$transform(dtm_test)

#Predictions
test_pred_prob <- predict(
  model,
  dtm_test_tfidf,
  s = "lambda.min",
  type = "response"
)[,1]

test_pred_class <- ifelse(
  test_pred_prob > 0.5,
  "Relevant claim content",
  "N/A: No relevant content."
)

pred_df <- tibble(
  .id = df_test$.id,
  bclass.pred = test_pred_class
)

save(pred_df, file = "results/preds-group9.RData")

