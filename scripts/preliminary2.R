require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

source('scripts/preprocessing.R')

# path to activity files on repo
url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'

# load a few functions for the activity
source(paste(url, 'projection-functions.R', sep = ''))

# load raw data
load('data/claims-clean-example.RData')
load('data/proj_out.RData')

words_processed <- nlp_fn(claims_clean)

words_processed_data <- words_processed %>%
  select(-.id, -bclass)

words_labels <- words_processed %>%
  select(.id, bclass)

# find projections based on training data
proj_out <- projection_fn(.dtm = words_processed_data, .prop = 0.7)
dtm_projected <- proj_out$data

# create training matrix
train <- words_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(dtm_projected)

fit <- glm(bclass ~ ., data = train, family = binomial)


# compute predicted probabilities
word_predict <- predict(fit,  
                     newdata = dtm_projected,
                     type = 'link')

word_predict_link <- cbind(words_labels, word_predict)

bigram_processed <- nlp_fn_bigrams(claims_clean)

bigram_processed_data <- bigram_processed %>%
  select(-.id, -bclass)

bigram_labels <- bigram_processed %>%
  select(.id, bclass)

# find projections based on training data
proj_out_bigram <- projection_fn(.dtm = bigram_processed_data, .prop = 0.7)
dtm_projected <- proj_out_bigram$data