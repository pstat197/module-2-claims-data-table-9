## this script contains functions for preprocessing
## claims data; intended to be sourced 
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

# function to parse html and clean text
parse_fn <- function(.html){
  read_html(.html) %>%
    html_elements('p, h1, h2, h3, h4, h5, h6') %>%    # for paragraphs only: 'p', for paragraphs + headers: 'p, h1, h2, h3, h4, h5, h6'
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

# function to apply to claims data
parse_data <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}

nlp_fn <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

safe_parse_fn <- function(html) {
  if (is.na(html) || html == "") return(NA_character_)  # return NA for empty input
  tryCatch(
    parse_fn(html),  # call original parse_fn
    error = function(e) NA_character_  # return NA if parsing fails
  )
}

parse_data_safe_na <- function(df) {
  df %>%
    mutate(
      text_clean = map_chr(text_tmp, safe_parse_fn)  # always keeps the same number of rows
    )
}

parse_data_safe <- function(df, chunk_size = 50) {
  n <- nrow(df)
  out_list <- list()
  
  for (i in seq(1, n, by = chunk_size)) {
    idx <- i:min(i + chunk_size - 1, n)
    chunk <- df[idx, ]
    
    # Only parse rows with HTML
    chunk <- chunk %>% filter(str_detect(text_tmp, '<!'))
    
    # Use safe_parse_fn instead of parse_fn
    chunk$text_clean <- map_chr(chunk$text_tmp, safe_parse_fn)
    
    out_list[[length(out_list) + 1]] <- chunk
    
    # Free memory after each chunk
    rm(chunk)
    gc()
  }
  
  bind_rows(out_list)
}

nlp_fn_safe <- function(df, chunk_size = 50) {
  out_list <- list()
  n <- nrow(df)
  
  for (i in seq(1, n, by = chunk_size)) {
    idx <- i:min(i + chunk_size - 1, n)
    chunk <- df[idx, ]
    
    # Call your original nlp_fn but catch errors
    chunk_tokens <- tryCatch(
      nlp_fn(chunk), 
      error = function(e) {
        message("Chunk failed: ", i)
        return(NULL)
      }
    )
    
    if (!is.null(chunk_tokens)) {
      out_list[[length(out_list) + 1]] <- chunk_tokens
    }
    
    rm(chunk, chunk_tokens)
    gc()
  }
  
  bind_rows(out_list)
}
