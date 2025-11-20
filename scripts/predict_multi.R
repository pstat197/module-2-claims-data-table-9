library(tm)
library(e1071)

# load model
load("results/svm_mclass_model.RData")   # loads svm_final

# load test data
load("data/claims-test.RData")
test_df <- parse_data_safe_na(claims_test)
test_df$text_clean[is.na(test_df$text_clean)] <- ""

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

# save
pred_df <- data.frame(
  .id = claims_test$.id,
  mclass.pred = mclass_pred
)
save(pred_df, file = "results/preds-multi-group9.RData")