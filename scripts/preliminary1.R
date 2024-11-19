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
load('data/claims-raw.RData')

# preprocess (will take a minute or two)
claims_clean_w_h <- claims_raw %>%
  parse_data_w_h()

# export
save(claims_clean_w_h, file = 'data/claims-clean-w-headers.RData')

# Start of 1
load('data/claims-clean-w-headers.RData')

data_processed <- nlp_fn(claims_clean_w_h)

# partition data
set.seed(102722)
partitions <- data_processed %>% initial_split(prop = 0.8)

# separate DTM from labels
test_dtm <- testing(partitions) %>%
  select(-.id, -bclass)
test_labels <- testing(partitions) %>%
  select(.id, bclass)

# same, training set
train_dtm <- training(partitions) %>%
  select(-.id, -bclass)
train_labels <- training(partitions) %>%
  select(.id, bclass)

# find projections based on training data
proj_out <- projection_fn(.dtm = train_dtm, .prop = 0.7)
train_dtm_projected <- proj_out$data

# how many components were used?
proj_out$n_pc

# create training matrix
train <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected)

fit <- glm(bclass ~ ., data = train, family = binomial)


# Comparing With Headers to Without
load('data/claims-clean-example.RData')

data_processed_n_h <- nlp_fn(claims_clean)

# partition data
set.seed(102722)
partitions_n_h <- data_processed_n_h %>% initial_split(prop = 0.8)

# separate DTM from labels
test_dtm_n_h <- testing(partitions_n_h) %>%
  select(-.id, -bclass)
test_labels_n_h <- testing(partitions_n_h) %>%
  select(.id, bclass)

# same, training set
train_dtm_n_h <- training(partitions_n_h) %>%
  select(-.id, -bclass)
train_labels_n_h <- training(partitions_n_h) %>%
  select(.id, bclass)

# find projections based on training data
proj_out_n_h <- projection_fn(.dtm = train_dtm_n_h, .prop = 0.7)
train_dtm_projected_n_h <- proj_out_n_h$data

# how many components were used?
proj_out_n_h$n_pc

# create training matrix
train_n_h <- train_labels_n_h %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm_projected_n_h)

fit_n_h <- glm(bclass ~ ., data = train_n_h, family = binomial)

