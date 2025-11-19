## Regenerating cleaned dataset with text_clean column

library(tidyverse)
library(rvest)
library(qdapRegex)
library(stopwords)
library(tidytext)
library(textstem)
library(tokenizers)

# Load preprocessing
source("scripts/preprocessing.R")

# Load raw HTML training data
load("data/claims-raw.RData")

#Parse HTML and clean text
clean_df <- parse_data(claims_raw)

claims_clean <- clean_df
save(claims_clean, file = "data/claims-clean.RData")
