library(tidyverse)
library(text2vec)
library(glmnet)

# Load training data
load("data/claims-clean.RData")
df <- claims_clean

# Binary outcome and dropping extra factor levels
df$bclass <- droplevels(df$bclass)
labels <- df$bclass
texts  <- df$text_clean

# Tokenization and TF-IDF w/ pruning
it <- itoken(texts,
             tokenizer = word_tokenizer,
             progressbar = TRUE)

#lower threshold and removing common words, 
#words that show up less than 3 times 
#removes words that appear in less than 50% of documents
vocab <- create_vocabulary(it) %>%
  prune_vocabulary(term_count_min = 3,    
                   doc_proportion_max = 0.5) 

vectorizer <- vocab_vectorizer(vocab)
dtm <- create_dtm(it, vectorizer)

tfidf <- TfIdf$new()
dtm_tfidf <- tfidf$fit_transform(dtm)



# LASSO logistic regression
y <- as.numeric(labels == levels(labels)[2])

# CV
set.seed(123)
cvfit <- cv.glmnet(dtm_tfidf, y,
                   family = "binomial",
                   alpha = 1,
                   type.measure = "class",
                   nfolds = 10)  
model <- cvfit

# test data
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
cv_accuracy <- 1 - model$cvm[ which(model$lambda == model$lambda.min) ]
cv_accuracy

# Training Data Performance
train_prob <- predict(
  model,
  dtm_tfidf,
  s = "lambda.min",
  type = "response"
)[,1]

train_pred <- ifelse(train_prob > 0.5, 1, 0)
true_y <- y

accuracy <- mean(train_pred == true_y)
sensitivity <- mean(train_pred[true_y == 1] == 1)
specificity <- mean(train_pred[true_y == 0] == 0)

accuracy
sensitivity
specificity
