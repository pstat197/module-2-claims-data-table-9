library(tidyverse)
library(rsample)
library(irlba)
library(Matrix)
library(dplyr)
library(glmnet)
library(quanteda)


source("scripts/preprocessing.R")

blpcr <- function(data_path) {
  # Load the .RData file
  load(data_path)
  
  # Ensure the loaded data is accessible (assumes it is called claims_clean)
  data_processed <- parse_data(claims_clean)
  data_more <- nlp_fn(data_processed)
  
  # Create binary response variable
  data_more$bclass <- ifelse(data_more$bclass == "Relevant claim content", 1, 0)
  
  # Separate predictors and response
  X <- as.matrix(data_more[, -1])
  y <- data_more$bclass
  
  # Scale predictors
  X_scaled <- scale(X)
  
  # Perform PCA
  pca <- prcomp(X_scaled, center = TRUE, scale. = TRUE)
  X_pcs <- pca$x[, 1:k]  # First k principal components
  
  # Split into train/test (80/20 split)
  set.seed(123)
  n <- nrow(X_pcs)
  train_idx <- sample(seq_len(n), size = 0.8 * n)
  X_train <- X_pcs[train_idx, ]
  y_train <- y[train_idx]
  X_test <- X_pcs[-train_idx, ]
  y_test <- y[-train_idx]
  
  # Fit logistic regression on principal components
  model <- glm(y_train ~ ., data = as.data.frame(X_train), family = binomial)
  
  # Predict on test set
  probs <- predict(model, newdata = as.data.frame(X_test), type = "response")
  preds <- ifelse(probs > 0.5, 1, 0)
  
  # Calculate accuracy
  accuracy <- mean(preds == y_test)
  
  return(accuracy)

}

# 1. Accuracy WITHOUT headers
#accuracy_no_headers <- blpcr("data/claims-clean-paragraphs.RData")  #paragraphs only
#print(paste("Accuracy without headers:", accuracy_no_headers))
#Accuracy without headers: 0.55952380952381


# 2. Accuracy WITH headers
#accuracy_headers <- blpcr("data/claims-clean-paragraphs_headers.RData")  #headers + paragraphs
#print(paste("Accuracy with headers:", accuracy_headers))
#Accuracy with headers: 0.546835443037975

##Task 2

blpcr_stacked <- function(data_path, k_unigram = 20, k_bigram = 20) {
  # Load data
  load(data_path)
  
  # Step 1: Preprocess
  data_parsed <- parse_data(claims_clean)
  
  # --- UNIGRAM MODEL ---
  data_uni <- nlp_fn(data_parsed)
  data_uni$bclass <- ifelse(data_uni$bclass == "Relevant claim content", 1, 0)
  
  X_uni <- as.matrix(data_uni[, -1])  # Drop .id
  y <- data_uni$bclass
  
  # Split indices for fair evaluation
  set.seed(123)
  n <- nrow(X_uni)
  train_idx <- sample(seq_len(n), size = 0.8 * n)
  test_idx <- setdiff(seq_len(n), train_idx)
  
  # PCA for unigrams
  X_uni_scaled <- scale(X_uni)
  pca_uni <- prcomp(X_uni_scaled)
  X_uni_pcs <- pca_uni$x[, 1:k_unigram]
  
  # Fit logistic PCR model on unigram PCs
  model_uni <- glm(y[train_idx] ~ ., data = as.data.frame(X_uni_pcs[train_idx, ]), family = binomial)
  
  # Predict log-odds on test set
  log_odds_test <- predict(model_uni, newdata = as.data.frame(X_uni_pcs[test_idx, ]), type = "link")
  
  # --- BIGRAM FEATURES ---
  nlp_fn_bigrams_topk <- function(parse_data.out, top_k = 5000) {
    
    # Build corpus
    corp <- corpus(parse_data.out, text_field = "text_clean")
    
    # Bigram tokens
    toks <- tokens(corp, what = "word") %>%
      tokens_ngrams(n = 2)
    
    # Initial dfm (sparse)
    dfm_bi <- dfm(toks)
    
    # --- NEW STEP: TRIM VOCABULARY ---
    # Keep only the top-k most frequent bigrams
    dfm_bi_trim <- dfm_trim(dfm_bi, max_features = top_k)
    
    # Compute TF–IDF (still sparse)
    dfm_tfidf_bi <- dfm_tfidf(dfm_bi_trim)
    
    # Convert to sparse Matrix
    m <- quanteda::convert(dfm_tfidf_bi, to = "Matrix")
    
    # Return sparse matrix + metadata
    out <- list(
      X = m,
      y = parse_data.out$bclass,
      ids = parse_data.out$.id
    )
    
    return(out)
  }
  
  data_bi <- nlp_fn_bigrams(data_parsed)
  data_bi$bclass <- ifelse(data_bi$bclass == "Relevant claim content", 1, 0)
  
  X_bi <- as.matrix(data_bi[, -1])  # Drop .id
  X_bi_scaled <- scale(X_bi)
  pca_bi <- prcomp(X_bi_scaled)
  X_bi_pcs <- pca_bi$x[, 1:k_bigram]
  
  # Get bigram PCs for test set
  X_bi_test <- X_bi_pcs[test_idx, ]
  
  # Stack: log-odds + bigram PCs
  X_stack <- cbind(log_odds_test, X_bi_test)
  
  # Fit second-stage logistic model
  model_stack <- glm(y[test_idx] ~ ., data = as.data.frame(X_stack), family = binomial)
  
  # Predict probabilities & classes
  probs <- predict(model_stack, type = "response")
  preds <- ifelse(probs > 0.5, 1, 0)
  
  # Accuracy
  accuracy <- mean(preds == y[test_idx])
  return(accuracy)
}

acc_stacked <- blpcr_stacked("data/claims-clean-paragraphs.RData")
print(paste("Stacked bigram + log-odds accuracy:", acc_stacked))
#Stacked bigram + log-odds accuracy: 0.58324321984908
