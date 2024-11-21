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

# Removing url2328 since the text_clean is just one word
claims_clean <- claims_clean %>%
  filter(.id != "url2328")

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

word_predict_df <- cbind(words_labels, word_predict)
  
# Lemmatize both words in the bigram
bigrams_words <- nlp_fn_bigrams(claims_clean)

# separate DTM from labels
data_dtm <- bigrams_words %>%
  select(-.id, -bclass)
data_labels <- bigrams_words %>%
  select(.id, bclass)

# find projections based on training data
proj_out_bigrams <- projection_fn(.dtm = data_dtm, .prop = 0.7)
dtm_projected_bigrams <- proj_out_bigrams$data

# create training matrix
bigrams_data <- data_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(dtm_projected_bigrams) %>%
  bind_cols(word_predict)

# partition data
set.seed(102722)
partition <- bigrams_data %>% initial_split(prop = 0.8)

# separate DTM from labels
test_dtm <- testing(partition) %>%
  select(-bclass)
test_labels <- testing(partition) %>%
  select(bclass)

# same, training set
train_dtm <- training(partition) %>%
  select(-bclass)
train_labels <- training(partition) %>%
  select(bclass)

# create training matrix
train_df <- train_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(train_dtm)

fit_bigram <- glm(bclass ~ ., data = train_df, family = binomial)

save(fit_bigram, file = 'results/binomial-fit-with-bigram.RData')

# compute predicted probabilities
preds <- predict(fit_bigram,  
                     newdata = test_dtm,
                     type = 'response')

# store predictions in a data frame with true labels
pred <- test_labels %>%
  transmute(bclass = factor(bclass)) %>%
  bind_cols(pred = as.numeric(preds)) %>%
  mutate(bclass.pred = factor(pred > 0.5, 
                                  labels = levels(bclass)))

panel <- metric_set(sensitivity, 
                    specificity, 
                    accuracy, 
                    roc_auc)

# compute test set accuracy for with headers
metrics_with_bigrams <- pred %>% panel(truth = bclass, 
                  estimate = bclass.pred, 
                  pred, 
                  event_level = 'second')

save(metrics_with_bigrams, file = 'data/metrics-with-bigrams.RData')

in_words <- words_labels %>%
  anti_join(bigrams_words_labels, by = '.id')
